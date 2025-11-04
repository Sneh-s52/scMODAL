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
from typing import Tuple, Optional

class DeepCCA(nn.Module):
    """
    Deep Canonical Correlation Analysis (Deep CCA) module
    """
    def __init__(self, latent_dim: int, cca_dim: int, hidden_dims: list = [64, 32]):
        super(DeepCCA, self).__init__()
        self.latent_dim = latent_dim
        self.cca_dim = cca_dim
        self.hidden_dims = hidden_dims
        
        # View-specific projection networks
        self.projection_A = self._build_projection_network()
        self.projection_B = self._build_projection_network()
        
    def _build_projection_network(self) -> nn.Sequential:
        layers = []
        input_dim = self.latent_dim
        
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.cca_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, z_A: torch.Tensor, z_B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Deep CCA
        
        Args:
            z_A: Latent representation from modality A [batch_size, latent_dim]
            z_B: Latent representation from modality B [batch_size, latent_dim]
            
        Returns:
            Tuple of (projected_A, projected_B) in CCA space
        """
        projected_A = self.projection_A(z_A)
        projected_B = self.projection_B(z_B)
        
        return projected_A, projected_B

def deep_cca_loss(projected_A: torch.Tensor, projected_B: torch.Tensor, 
                  reg_param: float = 1e-4) -> torch.Tensor:
    """
    Compute Deep CCA loss using correlation maximization
    
    Args:
        projected_A: Projected representations from modality A
        projected_B: Projected representations from modality B
        reg_param: Regularization parameter for numerical stability
        
    Returns:
        CCA loss (negative correlation to be minimized)
    """
    batch_size = projected_A.size(0)
    
    # Center the projections
    projected_A_centered = projected_A - projected_A.mean(dim=0, keepdim=True)
    projected_B_centered = projected_B - projected_B.mean(dim=0, keepdim=True)
    
    # Compute covariance matrices
    cov_AA = (projected_A_centered.T @ projected_A_centered) / (batch_size - 1)
    cov_BB = (projected_B_centered.T @ projected_B_centered) / (batch_size - 1)
    cov_AB = (projected_A_centered.T @ projected_B_centered) / (batch_size - 1)
    
    # Add regularization for numerical stability
    cov_AA_reg = cov_AA + reg_param * torch.eye(cov_AA.size(0), device=cov_AA.device)
    cov_BB_reg = cov_BB + reg_param * torch.eye(cov_BB.size(0), device=cov_BB.device)
    
    # Compute the CCA objective (correlation)
    try:
        # Compute the matrix square roots using SVD for stability
        U_A, S_A, V_A = torch.svd(cov_AA_reg)
        U_B, S_B, V_B = torch.svd(cov_BB_reg)
        
        # Whitening matrices
        whitening_A = U_A @ torch.diag(1.0 / torch.sqrt(S_A)) @ V_A.T
        whitening_B = U_B @ torch.diag(1.0 / torch.sqrt(S_B)) @ V_B.T
        
        # Compute correlation
        correlation_matrix = whitening_A @ cov_AB @ whitening_B.T
        correlation = torch.trace(correlation_matrix @ correlation_matrix.T)
        
    except RuntimeError:
        # Fallback if SVD fails
        correlation = torch.tensor(0.0, device=projected_A.device)
    
    # Return negative correlation (since we want to maximize correlation)
    return -correlation

def compute_cca_alignment(z_A: torch.Tensor, z_B: torch.Tensor, 
                         cca_dim: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute CCA-aligned representations using traditional CCA
    
    Args:
        z_A: Latent representation from modality A
        z_B: Latent representation from modality B
        cca_dim: Dimension for CCA projection (default: min(latent_dim, batch_size-1))
        
    Returns:
        Tuple of (aligned_A, aligned_B) after CCA alignment
    """
    batch_size, latent_dim = z_A.shape
    
    if cca_dim is None:
        cca_dim = min(latent_dim, batch_size - 1)
    
    # Convert to numpy for traditional CCA computation
    z_A_np = z_A.detach().cpu().numpy()
    z_B_np = z_B.detach().cpu().numpy()
    
    # Center the data
    z_A_centered = z_A_np - z_A_np.mean(axis=0, keepdims=True)
    z_B_centered = z_B_np - z_B_np.mean(axis=0, keepdims=True)
    
    # Compute covariance matrices
    cov_AA = z_A_centered.T @ z_A_centered / (batch_size - 1)
    cov_BB = z_B_centered.T @ z_B_centered / (batch_size - 1)
    cov_AB = z_A_centered.T @ z_B_centered / (batch_size - 1)
    
    # Add regularization
    reg_param = 1e-6
    cov_AA_reg = cov_AA + reg_param * np.eye(cov_AA.shape[0])
    cov_BB_reg = cov_BB + reg_param * np.eye(cov_BB.shape[0])
    
    # Compute CCA transformation matrices
    try:
        # Compute matrix square roots
        sqrt_AA = np.linalg.cholesky(cov_AA_reg)
        sqrt_BB = np.linalg.cholesky(cov_BB_reg)
        
        # Compute CCA matrix
        C = np.linalg.inv(sqrt_AA) @ cov_AB @ np.linalg.inv(sqrt_BB.T)
        
        # SVD to get transformation matrices
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        
        # Take top cca_dim components
        U = U[:, :cca_dim]
        Vt = Vt[:cca_dim, :]
        S = S[:cca_dim]
        
        # Transformation matrices
        transform_A = np.linalg.inv(sqrt_AA) @ U
        transform_B = np.linalg.inv(sqrt_BB) @ Vt.T
        
        # Apply transformations
        aligned_A_np = z_A_centered @ transform_A
        aligned_B_np = z_B_centered @ transform_B
        
        # Convert back to torch tensors
        aligned_A = torch.from_numpy(aligned_A_np).float().to(z_A.device)
        aligned_B = torch.from_numpy(aligned_B_np).float().to(z_B.device)
        
        return aligned_A, aligned_B
        
    except np.linalg.LinAlgError:
        # Fallback: return original representations
        print("CCA computation failed, returning original representations")
        return z_A, z_B

def integrate_deep_cca_with_gan(z_A: torch.Tensor, z_B: torch.Tensor, 
                               deep_cca_module: DeepCCA,
                               lambda_cca: float = 1.0,
                               use_cca_alignment: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate Deep CCA with GAN by aligning representations before GAN mixing
    
    Args:
        z_A: Latent representation from modality A
        z_B: Latent representation from modality B
        deep_cca_module: Deep CCA module
        lambda_cca: Weight for CCA loss
        use_cca_alignment: Whether to use CCA-aligned representations for GAN
        
    Returns:
        Tuple of (z_A_mixed, z_B_mixed, cca_loss)
    """
    # Apply Deep CCA
    projected_A, projected_B = deep_cca_module(z_A, z_B)
    
    # Compute CCA loss
    cca_loss = deep_cca_loss(projected_A, projected_B)
    
    if use_cca_alignment:
        # Use CCA-aligned representations for GAN mixing
        z_A_aligned, z_B_aligned = compute_cca_alignment(projected_A, projected_B)
        
        # Mix representations (you can customize this mixing strategy)
        z_A_mixed = 0.7 * z_A + 0.3 * z_A_aligned
        z_B_mixed = 0.7 * z_B + 0.3 * z_B_aligned
    else:
        # Use original representations
        z_A_mixed = z_A
        z_B_mixed = z_B
    
    return z_A_mixed, z_B_mixed, cca_loss
                                   
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

