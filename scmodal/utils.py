import os
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad
import umap
from annoy import AnnoyIndex
from scmodal.model import *
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepCCA(nn.Module):
    """
    Deep Canonical Correlation Analysis (Deep CCA) layer
    Takes encoder latent representations and learns correlated features
    """
    def __init__(self, input_dim1, input_dim2, latent_dim, hidden_dims=[512, 256]):
        super(DeepCCA, self).__init__()
        self.latent_dim = latent_dim
        
        # Transformation network for modality A (encoder latent -> DCCA latent)
        transform_layers_A = []
        prev_dim = input_dim1
        for hidden_dim in hidden_dims:
            transform_layers_A.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        transform_layers_A.append(nn.Linear(prev_dim, latent_dim))
        self.transform_A = nn.Sequential(*transform_layers_A)
        
        # Transformation network for modality B (encoder latent -> DCCA latent)
        transform_layers_B = []
        prev_dim = input_dim2
        for hidden_dim in hidden_dims:
            transform_layers_B.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        transform_layers_B.append(nn.Linear(prev_dim, latent_dim))
        self.transform_B = nn.Sequential(*transform_layers_B)
        
    def forward(self, z1, z2):
        z1_dcca = self.transform_A(z1)
        z2_dcca = self.transform_B(z2)
        return z1_dcca, z2_dcca


def deep_cca_loss(z1_encoder, z2_encoder, z1_dcca, z2_dcca, r1=0.1, r2=0.1):  # Much stronger regularization
    """
    Conservative DCCA loss to prevent over-regularization
    """
    batch_size, latent_dim = z1_dcca.shape
    
    # Center the embeddings
    z1_dcca_centered = z1_dcca - z1_dcca.mean(dim=0)
    z2_dcca_centered = z2_dcca - z2_dcca.mean(dim=0)
    
    # Simple correlation-based loss (more stable)
    corr_matrix = (z1_dcca_centered.T @ z2_dcca_centered) / (batch_size - 1)
    
    # Use Frobenius norm with strong regularization
    correlation_strength = torch.norm(corr_matrix, p='fro')
    
    # Return negative correlation (to maximize) but scaled down
    return -correlation_strength * 0.1  # Small scaling factor

def acquire_pairs(X, Y, k=30, metric='angular'):
    # This function was modified from iMAP: https://github.com/Svvord/iMAP/blob/master/imap/stage2.py
    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool_(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat).astype(int)
    return mnn_mat
     
def annotate_by_nn(vec_tar, vec_ref, label_ref, k=20, metric='cosine'):
    dist_mtx = cdist(vec_tar, vec_ref, metric=metric)
    idx = dist_mtx.argsort()[:, :k]
    labels = [max(list(label_ref[i]), key=list(label_ref[i]).count) for i in idx]
    return labels

def compute_umap(adata, rep=None):
    import umap

    reducer = umap.UMAP(n_neighbors=30,
                        n_components=2,
                        metric="correlation",
                        n_epochs=None,
                        learning_rate=1.0,
                        min_dist=0.3,
                        spread=1.0,
                        set_op_mix_ratio=1.0,
                        local_connectivity=1,
                        repulsion_strength=1,
                        negative_sample_rate=5,
                        a=None,
                        b=None,
                        random_state=1234,
                        metric_kwds=None,
                        angular_rp_forest=False,
                        verbose=True)
    if rep is None:
        X_umap = reducer.fit_transform(adata.X)
    else:
        X_umap = reducer.fit_transform(adata.obsm[rep])

    adata.obsm['X_umap'] = X_umap
