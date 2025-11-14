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
    
    Based on: Andrew et al. "Deep Canonical Correlation Analysis", ICML 2013
    Reference implementation: https://github.com/VahidooX/DeepCCA
    
    Takes encoder latent representations and learns correlated features by
    maximizing the sum of canonical correlations between the two modalities.
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
                nn.LayerNorm(hidden_dim),  # More stable than BatchNorm for variable batch sizes
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
                nn.LayerNorm(hidden_dim),  # More stable than BatchNorm for variable batch sizes
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


def deep_cca_loss(z1_encoder, z2_encoder, z1_dcca, z2_dcca, r1=1e-4, r2=1e-4, use_all_singular_values=True):
    """
    Deep CCA loss function that maximizes canonical correlations.
    
    Based on the standard DCCA formulation from:
    - Andrew et al. "Deep Canonical Correlation Analysis", ICML 2013
    - Reference: https://github.com/VahidooX/DeepCCA
    
    The loss maximizes the trace of T = C11^(-1/2) * C12 * C22^(-1/2),
    which equals the sum of canonical correlations.
    
    Args:
        z1_encoder: Encoder output for modality 1 (not used in loss, kept for API compatibility)
        z2_encoder: Encoder output for modality 2 (not used in loss, kept for API compatibility)
        z1_dcca: DCCA-transformed output for modality 1
        z2_dcca: DCCA-transformed output for modality 2
        r1: Regularization parameter for covariance matrix of modality 1
        r2: Regularization parameter for covariance matrix of modality 2
        use_all_singular_values: If True, use all singular values; if False, use only top-k
    
    Returns:
        Negative trace (to maximize correlation via minimization)
    """
    batch_size, latent_dim = z1_dcca.shape
    
    # Center the embeddings (remove mean)
    z1_dcca_centered = z1_dcca - z1_dcca.mean(dim=0, keepdim=True)
    z2_dcca_centered = z2_dcca - z2_dcca.mean(dim=0, keepdim=True)
    
    # Compute covariance matrices with regularization
    # C11 = (1/N) * Z1^T * Z1 + r1 * I
    # C22 = (1/N) * Z2^T * Z2 + r2 * I
    # C12 = (1/N) * Z1^T * Z2
    C11 = (z1_dcca_centered.T @ z1_dcca_centered) / (batch_size - 1) + r1 * torch.eye(
        latent_dim, device=z1_dcca.device, dtype=z1_dcca.dtype
    )
    C22 = (z2_dcca_centered.T @ z2_dcca_centered) / (batch_size - 1) + r2 * torch.eye(
        latent_dim, device=z2_dcca.device, dtype=z2_dcca.dtype
    )
    C12 = (z1_dcca_centered.T @ z2_dcca_centered) / (batch_size - 1)
    
    # Compute T = C11^(-1/2) * C12 * C22^(-1/2)
    # The trace of T equals the sum of canonical correlations
    try:
        # Use Cholesky decomposition for numerical stability
        # C11 = L11 * L11^T, so C11^(-1/2) = L11^(-T)
        L11 = torch.linalg.cholesky(C11)
        L22 = torch.linalg.cholesky(C22)
        
        # Solve: L11 * X = C12  =>  X = L11^(-1) * C12
        X = torch.linalg.solve_triangular(L11, C12, upper=False)
        # Solve: L22^T * T = X^T  =>  T = (L22^(-T) * X^T)^T = X * L22^(-1)
        T = torch.linalg.solve_triangular(L22, X.T, upper=True).T
        
    except RuntimeError:
        # Fallback to SVD-based approach if Cholesky fails (shouldn't happen with regularization)
        # C11^(-1/2) = U1 * diag(1/sqrt(S1)) * V1^T
        U1, S1, V1 = torch.linalg.svd(C11)
        U2, S2, V2 = torch.linalg.svd(C22)
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        C11_inv_sqrt = U1 @ torch.diag_embed(1.0 / torch.sqrt(S1 + eps)) @ V1.T
        C22_inv_sqrt = U2 @ torch.diag_embed(1.0 / torch.sqrt(S2 + eps)) @ V2.T
        
        T = C11_inv_sqrt @ C12 @ C22_inv_sqrt
    
    # Compute the loss: maximize trace(T) = sum of canonical correlations
    if use_all_singular_values:
        # Use trace (sum of all canonical correlations)
        correlation = torch.trace(T)
    else:
        # Use only top-k singular values (more stable for high dimensions)
        # This is equivalent to using only the top-k canonical correlations
        singular_values = torch.linalg.svdvals(T)
        correlation = torch.sum(singular_values[:min(latent_dim, batch_size)])
    
    # Return negative correlation (to maximize via minimization)
    return -correlation

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
