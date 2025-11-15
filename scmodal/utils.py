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
    def __init__(
        self,
        input_dim1,
        input_dim2,
        latent_dim,
        hidden_dims=None,
        use_batch_norm=False,
        dropout=0.1,
    ):
        super(DeepCCA, self).__init__()
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [64, 32]
        hidden_dims = list(hidden_dims)
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout
        
        # Transformation network for modality A (encoder latent -> DCCA latent)
        self.transform_A = self._build_projection(input_dim1, hidden_dims, latent_dim)
        
        # Transformation network for modality B (encoder latent -> DCCA latent)
        self.transform_B = self._build_projection(input_dim2, hidden_dims, latent_dim)
    
    def _build_projection(self, input_dim, hidden_dims, output_dim):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if self.dropout and self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
        
    def forward(self, z1, z2):
        z1_dcca = self.transform_A(z1)
        z2_dcca = self.transform_B(z2)
        return z1_dcca, z2_dcca
    
    def freeze(self):
        """Freeze all DCCA parameters to stop gradient updates"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze all DCCA parameters to allow gradient updates"""
        for param in self.parameters():
            param.requires_grad = True


def deep_cca_loss(z1_encoder, z2_encoder, z1_dcca, z2_dcca, r1=1e-5, r2=1e-5, use_all_singular_values=True):
    """
    Deep CCA loss function that maximizes canonical correlations.
    
    Based on the standard DCCA formulation from:
    - Andrew et al. "Deep Canonical Correlation Analysis", ICML 2013
    - Reference: https://github.com/VahidooX/DeepCCA
    
    The loss maximizes the sum of singular values of T = C11^(-1/2) * C12 * C22^(-1/2),
    which equals the sum of canonical correlations. Note: canonical correlations are the
    singular values of T, NOT the diagonal elements (trace).
    
    Args:
        z1_encoder: Encoder output for modality 1 (not used in loss, kept for API compatibility)
        z2_encoder: Encoder output for modality 2 (not used in loss, kept for API compatibility)
        z1_dcca: DCCA-transformed output for modality 1
        z2_dcca: DCCA-transformed output for modality 2
        r1: Regularization parameter for covariance matrix of modality 1 (default: 1e-5)
        r2: Regularization parameter for covariance matrix of modality 2 (default: 1e-5)
        use_all_singular_values: If True, use all singular values; if False, use only top-k
    
    Returns:
        Negative sum of singular values scaled by 1/latent_dim (to maximize correlation via minimization)
        The scaling ensures the loss magnitude is comparable to other losses.
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
    
    # Compute T = C11^(-1/2) * C12 * C22^(-1/2) using SVD for numerical stability
    # This is more stable than Cholesky for the matrix square root
    eps = 1e-8
    
    # Compute C11^(-1/2) using SVD: C11 = U1 * S1 * V1^T, so C11^(-1/2) = U1 * diag(1/sqrt(S1)) * V1^T
    U1, S1, V1 = torch.linalg.svd(C11)
    # Ensure eigenvalues are positive (should be with regularization)
    S1 = torch.clamp(S1, min=eps)
    C11_inv_sqrt = U1 @ torch.diag_embed(1.0 / torch.sqrt(S1)) @ V1.T
    
    # Compute C22^(-1/2) similarly
    U2, S2, V2 = torch.linalg.svd(C22)
    S2 = torch.clamp(S2, min=eps)
    C22_inv_sqrt = U2 @ torch.diag_embed(1.0 / torch.sqrt(S2)) @ V2.T
    
    # Compute T = C11^(-1/2) * C12 * C22^(-1/2)
    T = C11_inv_sqrt @ C12 @ C22_inv_sqrt
    
    # Compute the loss: maximize sum of canonical correlations
    # CRITICAL: Canonical correlations are the SINGULAR VALUES of T, not the diagonal elements
    # The trace only equals the sum of singular values if T is diagonal (which it's not in general)
    singular_values = torch.linalg.svdvals(T)
    
    if use_all_singular_values:
        # Use sum of all singular values (all canonical correlations)
        correlation = torch.sum(singular_values)
    else:
        # Use only top-k singular values (more stable for high dimensions)
        # This is equivalent to using only the top-k canonical correlations
        correlation = torch.sum(singular_values[:min(latent_dim, batch_size)])
    
    # Return negative correlation scaled by 1/latent_dim to normalize the loss magnitude
    # This ensures the loss is on a similar scale regardless of latent dimension
    return -correlation / latent_dim

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
