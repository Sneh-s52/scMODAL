import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from scmodal.networks import *
from scmodal.utils import *

class Model(object):
    def __init__(self, batch_size=500, training_steps=10000, seed=1234, n_latent=20,
                 lambdaAE = 10.0, lambdaLA = 10.0, lambdaMNN = 1.0, lambdaGeo = 10.0, lambdaGAN = 1.0, n_KNN = 30,
                 model_path="models", data_path="data", result_path="results", 
                 use_harmony=True, harmony_max_iter_harmony=10, harmony_sigma=0.1, harmony_theta=2.0,
                 use_deep_cca=True, lambdaCCA=1.0, cca_dim=16, cca_hidden_dims=[64, 32], cca_mixing_ratio=0.3,
                 use_gat=False, gat_hidden=256, gat_heads=1, gat_dropout=0.1, gat_knn=10):

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.batch_size = batch_size
        self.training_steps = training_steps
        self.n_latent = n_latent
        self.lambdaAE = lambdaAE
        self.lambdaLA = lambdaLA
        self.lambdaMNN = lambdaMNN
        self.lambdaGeo = lambdaGeo
        self.lambdaGAN = lambdaGAN
        self.n_KNN = n_KNN
        self.model_path = model_path
        self.data_path = data_path
        self.result_path = result_path
        
        # Harmony parameters
        self.use_harmony = use_harmony
        self.harmony_max_iter_harmony = harmony_max_iter_harmony
        self.harmony_sigma = harmony_sigma
        self.harmony_theta = harmony_theta
        
        # Deep CCA parameters
        self.use_deep_cca = use_deep_cca
        self.lambdaCCA = lambdaCCA
        self.cca_dim = cca_dim
        self.cca_hidden_dims = cca_hidden_dims
        self.cca_mixing_ratio = cca_mixing_ratio
        
        # GAT parameters
        self.use_gat = use_gat
        self.gat_hidden = gat_hidden
        self.gat_heads = gat_heads
        self.gat_dropout = gat_dropout
        self.gat_knn = gat_knn
        
        # Precomputed Harmony embeddings
        self.precomputed_harmony_A = None
        self.precomputed_harmony_B = None
        
        # Deep CCA module (will be initialized during training)
        self.deep_cca = None
        
        # Check if harmonypy is available
        self.harmony_available = self._check_harmony_availability()

    def _check_harmony_availability(self):
        """Check if harmonypy is available and import it"""
        try:
            import harmonypy
            self.hm = harmonypy
            print("harmonypy successfully imported")
            return True
        except ImportError:
            print("harmonypy not available, using simple batch correction")
            return False

    def _simple_batch_correction(self, embeddings, batch_labels):
        """Simple but effective batch correction"""
        try:
            embeddings = np.array(embeddings, dtype=np.float32)
            batch_labels = np.array(batch_labels)
            
            unique_batches = np.unique(batch_labels)
            if len(unique_batches) <= 1:
                return embeddings
                
            # Method: Remove batch means
            batch_corrected = embeddings.copy()
            overall_mean = np.mean(embeddings, axis=0)
            
            for batch in unique_batches:
                batch_mask = batch_labels == batch
                batch_data = embeddings[batch_mask]
                
                if len(batch_data) > 0:
                    # Remove batch-specific mean
                    batch_mean = np.mean(batch_data, axis=0)
                    batch_corrected[batch_mask] = batch_data - batch_mean + overall_mean
            
            return batch_corrected
            
        except Exception as e:
            print(f"Simple batch correction failed: {e}")
            return embeddings

    def _precompute_harmony_embeddings(self):
        """Precompute Harmony embeddings once at the beginning"""
        if not self.use_harmony or not self.harmony_available:
            return
            
        try:
            print("Precomputing Harmony embeddings for entire dataset...")
            
            # Use shared features for Harmony
            if hasattr(self, 'feat_A_MNN') and hasattr(self, 'feat_B_MNN'):
                feat_A = self.feat_A_MNN
                feat_B = self.feat_B_MNN
            else:
                feat_A = self.emb_A[:, :self.shared_gene_num]
                feat_B = self.emb_B[:, :self.shared_gene_num]
            
            # Convert to dense if sparse
            if hasattr(feat_A, 'toarray'):
                feat_A = feat_A.toarray()
            if hasattr(feat_B, 'toarray'):
                feat_B = feat_B.toarray()
                
            combined_feats = np.vstack([feat_A, feat_B])
            combined_batches = np.concatenate([self.batch_labels_A, self.batch_labels_B])
            
            print(f"Precomputing Harmony on {len(combined_feats)} cells...")
            
            # Run Harmony once
            batch_str = [f"batch_{int(b)}" for b in combined_batches]
            meta_data = pd.DataFrame({'batch': batch_str})
            vars_use = ['batch']
            
            ho = self.hm.run_harmony(
                combined_feats, 
                meta_data, 
                vars_use,
                max_iter_harmony=self.harmony_max_iter_harmony,
                sigma=self.harmony_sigma,
                theta=self.harmony_theta
            )
            
            harmonized_all = ho.Z_corr.T
            
            # Split and store
            self.precomputed_harmony_A = harmonized_all[:len(feat_A)]
            self.precomputed_harmony_B = harmonized_all[len(feat_A):]
            
            print(f"Precomputed Harmony embeddings: A={self.precomputed_harmony_A.shape}, B={self.precomputed_harmony_B.shape}")
            
        except Exception as e:
            print(f"Precomputing Harmony failed: {e}")
            self.precomputed_harmony_A = None
            self.precomputed_harmony_B = None

    def _build_knn_adjacency(self, features, k=10):
        """Build k-NN adjacency matrix for GAT"""
        try:
            n_samples = features.shape[0]
            
            # Compute k-nearest neighbors
            knn = NearestNeighbors(n_neighbors=k, metric='cosine')
            knn.fit(features)
            distances, indices = knn.kneighbors(features)
            
            # Create adjacency matrix
            adj = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                adj[i, indices[i]] = 1
                adj[indices[i], i] = 1  # Symmetric
            
            # Normalize adjacency matrix
            rowsum = np.array(adj.sum(1))
            degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
            adj_normalized = degree_mat_inv_sqrt.dot(adj).dot(degree_mat_inv_sqrt)
            
            return torch.FloatTensor(adj_normalized).to(self.device)
            
        except Exception as e:
            print(f"KNN adjacency failed: {e}")
            # Return identity matrix as fallback
            return torch.eye(features.shape[0]).to(self.device)

    def _get_gat_adjacency_batch(self, features_batch, k=10):
        """Build adjacency matrix for a batch of features"""
        try:
            if features_batch.shape[0] <= k:
                # If batch is smaller than k, use fully connected
                adj = torch.ones(features_batch.shape[0], features_batch.shape[0])
            else:
                # Convert to numpy for kNN
                features_np = features_batch.detach().cpu().numpy()
                adj = self._build_knn_adjacency(features_np, k)
            
            return adj.to(self.device)
        except Exception as e:
            print(f"Batch adjacency failed: {e}")
            return torch.eye(features_batch.shape[0]).to(self.device)

    def _get_harmonized_mnn_pairs_fast(self, index_A, index_B):
        """Fast MNN pairs using precomputed Harmony embeddings"""
        if not self.use_harmony or self.precomputed_harmony_A is None or self.precomputed_harmony_B is None:
            # Fallback to simple method
            if hasattr(self, 'feat_A_MNN') and hasattr(self, 'feat_B_MNN'):
                feat_A_batch = self.feat_A_MNN[index_A]
                feat_B_batch = self.feat_B_MNN[index_B]
            else:
                feat_A_batch = self.emb_A[index_A, :self.shared_gene_num]
                feat_B_batch = self.emb_B[index_B, :self.shared_gene_num]
                
            if hasattr(feat_A_batch, 'toarray'):
                feat_A_batch = feat_A_batch.toarray()
            if hasattr(feat_B_batch, 'toarray'):
                feat_B_batch = feat_B_batch.toarray()
                
            return acquire_pairs(feat_A_batch, feat_B_batch, k=self.n_KNN)
        
        # Use precomputed Harmony embeddings
        harmony_A_batch = self.precomputed_harmony_A[index_A]
        harmony_B_batch = self.precomputed_harmony_B[index_B]
        
        return acquire_pairs(harmony_A_batch, harmony_B_batch, k=self.n_KNN)

    def _get_harmonized_mnn_pairs_multi(self, feat_A, feat_B, batch_labels_A, batch_labels_B):
        """Get MNN pairs for multi-dataset case using simple batch correction"""
        if not self.use_harmony:
            return acquire_pairs(feat_A, feat_B, k=self.n_KNN)
            
        try:
            # Convert to dense arrays if sparse
            if hasattr(feat_A, 'toarray'):
                feat_A = feat_A.toarray()
            if hasattr(feat_B, 'toarray'):
                feat_B = feat_B.toarray()
                
            feat_A = np.array(feat_A, dtype=np.float32)
            feat_B = np.array(feat_B, dtype=np.float32)
            
            combined_feats = np.vstack([feat_A, feat_B])
            combined_batches = np.concatenate([batch_labels_A, batch_labels_B])
            
            # Use simple batch correction for multi-dataset (faster than full Harmony)
            corrected_feats = self._simple_batch_correction(combined_feats, combined_batches)
            
            corrected_A = corrected_feats[:len(feat_A)]
            corrected_B = corrected_feats[len(feat_A):]
            
            return acquire_pairs(corrected_A, corrected_B, k=self.n_KNN)
            
        except Exception as e:
            print(f"Multi-dataset batch-corrected MNN failed: {e}")
            return acquire_pairs(feat_A, feat_B, k=self.n_KNN)

    def _get_harmonized_mnn_pairs(self, feat_A, feat_B, batch_labels_A, batch_labels_B):
        """Backward compatibility method - redirects to the new fast method"""
        # For multi-dataset case, we need to handle it differently
        # Since we don't have precomputed embeddings for arbitrary pairs
        return self._get_harmonized_mnn_pairs_multi(feat_A, feat_B, batch_labels_A, batch_labels_B)

    def _apply_deep_cca_alignment(self, z_A, z_B):
        """Apply Deep CCA alignment to latent representations"""
        if not self.use_deep_cca or self.deep_cca is None:
            return z_A, z_B, torch.tensor(0.0).to(self.device)
        
        try:
            # Apply Deep CCA projection
            projected_A, projected_B = self.deep_cca(z_A, z_B)
            
            # Compute CCA loss
            cca_loss = deep_cca_loss(projected_A, projected_B)
            
            # Compute CCA-aligned representations using traditional CCA
            z_A_aligned, z_B_aligned = compute_cca_alignment(projected_A, projected_B, cca_dim=self.cca_dim)
            
            # Mix original and CCA-aligned representations
            mixing_ratio = self.cca_mixing_ratio
            z_A_mixed = (1 - mixing_ratio) * z_A + mixing_ratio * z_A_aligned
            z_B_mixed = (1 - mixing_ratio) * z_B + mixing_ratio * z_B_aligned
            
            return z_A_mixed, z_B_mixed, cca_loss
            
        except Exception as e:
            print(f"Deep CCA alignment failed: {e}")
            return z_A, z_B, torch.tensor(0.0).to(self.device)

    def preprocess(self, 
                   adata_A_input, 
                   adata_B_input, 
                   shared_gene_num
                   ):
        self.adata_A = adata_A_input.copy()
        self.adata_B = adata_B_input.copy()

        self.shared_gene_num = shared_gene_num
        self.emb_A = self.adata_A.X
        self.emb_B = self.adata_B.X
        
        # Create batch labels for Harmony
        self.batch_labels_A = np.zeros(self.emb_A.shape[0])
        self.batch_labels_B = np.ones(self.emb_B.shape[0])
        
        # Precompute Harmony embeddings
        self._precompute_harmony_embeddings()

    def preprocess_additional_inputs(self, 
                   adata_A_input, 
                   adata_B_input, 
                   shared_gene_num,
                   layer_adata_A_MNN=None, 
                   layer_adata_B_MNN=None, 
                   ):
        assert ((layer_adata_A_MNN is not None) or (layer_adata_B_MNN is not None)), "One of the layer names should be feeded; otherwise, use .preprocess() function."
        adata_A = adata_A_input.copy()
        adata_B = adata_B_input.copy()

        self.shared_gene_num = shared_gene_num
        self.emb_A = adata_A.X
        self.emb_B = adata_B.X
        
        # Create batch labels for Harmony
        self.batch_labels_A = np.zeros(self.emb_A.shape[0])
        self.batch_labels_B = np.ones(self.emb_B.shape[0])
        
        if layer_adata_A_MNN is None:
            self.feat_A_MNN = self.emb_A
        else:
            self.feat_A_MNN = adata_A.obsm[layer_adata_A_MNN]
        if layer_adata_B_MNN is None:
            self.feat_B_MNN = self.emb_B
        else:
            self.feat_B_MNN = adata_B.obsm[layer_adata_B_MNN]
            
        # Precompute Harmony embeddings
        self._precompute_harmony_embeddings()

    def train(self):
        begin_time = time.time()
        print("Beginning time: ", time.asctime(time.localtime(begin_time)))
        print(f"Using Harmony for MNN: {self.use_harmony}")
        print(f"Using Deep CCA: {self.use_deep_cca}")
        print(f"Using GAT: {self.use_gat}")
        
        # Show precomputation status
        if self.use_harmony:
            if self.precomputed_harmony_A is not None:
                print("✓ Using precomputed Harmony embeddings")
            else:
                print("✗ Using fallback batch correction")
        
        # Initialize encoders with GAT parameter
        self.E_A = encoder(self.emb_A.shape[1], self.n_latent, 
                          use_gat=self.use_gat, 
                          gat_hidden=self.gat_hidden,
                          gat_heads=self.gat_heads,
                          dropout=self.gat_dropout).to(self.device)
        self.E_B = encoder(self.emb_B.shape[1], self.n_latent,
                          use_gat=self.use_gat,
                          gat_hidden=self.gat_hidden,
                          gat_heads=self.gat_heads,
                          dropout=self.gat_dropout).to(self.device)
        
        self.G_A = generator(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.G_B = generator(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.D_Z = discriminator(self.n_latent).to(self.device)
        
        # Initialize Deep CCA if enabled
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        if self.use_deep_cca:
            self.deep_cca = DeepCCA(
                latent_dim=self.n_latent,
                cca_dim=self.cca_dim,
                hidden_dims=self.cca_hidden_dims
            ).to(self.device)
            params_G += list(self.deep_cca.parameters())
            print(f"Deep CCA initialized: latent_dim={self.n_latent}, cca_dim={self.cca_dim}")
        
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)
        optimizer_D = optim.Adam(list(self.D_Z.parameters()), lr=0.001, weight_decay=0.001)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_Z.train()
        if self.use_deep_cca:
            self.deep_cca.train()

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            
            # Get original latent representations with GAT if enabled
            if self.use_gat:
                adj_A = self._get_gat_adjacency_batch(x_A, k=self.gat_knn)
                adj_B = self._get_gat_adjacency_batch(x_B, k=self.gat_knn)
                z_A = self.E_A(x_A, adj_A)
                z_B = self.E_B(x_B, adj_B)
            else:
                z_A = self.E_A(x_A)
                z_B = self.E_B(x_B)
            
            # Apply Deep CCA alignment BEFORE GAN mixing
            z_A_aligned, z_B_aligned, loss_CCA = self._apply_deep_cca_alignment(z_A, z_B)
            
            # Use aligned representations for subsequent computations
            x_AtoB = self.G_B(z_A_aligned)
            x_BtoA = self.G_A(z_B_aligned)
            x_Arecon = self.G_A(z_A_aligned)
            x_Brecon = self.G_B(z_B_aligned)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)
            
            # Use aligned representations for discriminator
            z_A_for_disc = z_A_aligned
            z_B_for_disc = z_B_aligned
            
            K_A = torch.mean((x_A.view(self.batch_size, 1, -1) - x_A.view(1, self.batch_size, -1))**2, dim=2)
            K_A = torch.exp(-K_A/2)
            K_B_z = torch.mean((z_B_for_disc.view(self.batch_size, 1, -1) - z_B_for_disc.view(1, self.batch_size, -1))**2, dim=2)
            K_B_z = torch.exp(-K_B_z/2)
            K_B = torch.mean((x_B.view(self.batch_size, 1, -1) - x_B.view(1, self.batch_size, -1))**2, dim=2)
            K_B = torch.exp(-K_B/2)
            K_A_z = torch.mean((z_A_for_disc.view(self.batch_size, 1, -1) - z_A_for_disc.view(1, self.batch_size, -1))**2, dim=2)
            K_A_z = torch.exp(-K_A_z/2)

            # discriminator loss:
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = (torch.log(1 + torch.exp(-self.D_Z(z_A_for_disc))) + torch.log(1 + torch.exp(self.D_Z(z_B_for_disc)))).mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss:
            loss_LA_AtoB = torch.mean((z_A_aligned - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B_aligned - z_BtoA)**2)
            loss_LA = loss_LA_AtoB + loss_LA_BtoA

            # generator loss
            loss_G_GAN = -(torch.log(1 + torch.exp(-self.D_Z(z_A_for_disc))) + torch.log(1 + torch.exp(self.D_Z(z_B_for_disc)))).mean()

            # geometric structure loss
            loss_Geo = - (torch.clamp(cos(K_A, K_A_z), max=0.975).mean() + torch.clamp(cos(K_B, K_B_z), max=0.975).mean())

            # MNN loss - FAST VERSION USING PRECOMPUTED EMBEDDINGS
            Sim = self._get_harmonized_mnn_pairs_fast(index_A, index_B)
            Sim = torch.from_numpy(Sim).float().to(self.device)
            z_dist = torch.mean((z_A_aligned.view(self.batch_size, 1, -1) - z_B_aligned.view(1, self.batch_size, -1))**2, dim=2)
            loss_MNN = torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = (self.lambdaGAN * loss_G_GAN + 
                     self.lambdaAE * loss_AE + 
                     self.lambdaLA * loss_LA + 
                     self.lambdaMNN * loss_MNN + 
                     self.lambdaGeo * loss_Geo +
                     self.lambdaCCA * loss_CCA)  # Add CCA loss
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 2000:
                harmony_status = " (Harmony)" if self.use_harmony else ""
                cca_status = " (Deep CCA)" if self.use_deep_cca else ""
                gat_status = " (GAT)" if self.use_gat else ""
                print(f"step {step}{harmony_status}{cca_status}{gat_status}, loss_D={loss_D:.6f}, loss_GAN={loss_G_GAN:.6f}, loss_AE={self.lambdaAE*loss_AE:.6f}, loss_Geo={self.lambdaGeo*loss_Geo:.6f}, loss_LA={self.lambdaLA*loss_LA:.6f}, loss_MNN={self.lambdaMNN*loss_MNN:.6f}, loss_CCA={self.lambdaCCA*loss_CCA:.6f}")

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict()}
        
        # Save Deep CCA state if used
        if self.use_deep_cca:
            state['deep_cca'] = self.deep_cca.state_dict()

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))

    def eval(self):
        begin_time = time.time()
        print("Beginning time: ", time.asctime(time.localtime(begin_time)))

        self.E_A = encoder(self.emb_A.shape[1], self.n_latent, 
                          use_gat=self.use_gat, 
                          gat_hidden=self.gat_hidden,
                          gat_heads=self.gat_heads,
                          dropout=self.gat_dropout).to(self.device)
        self.E_B = encoder(self.emb_B.shape[1], self.n_latent,
                          use_gat=self.use_gat,
                          gat_hidden=self.gat_hidden,
                          gat_heads=self.gat_heads,
                          dropout=self.gat_dropout).to(self.device)
        self.G_A = generator(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.G_B = generator(self.emb_B.shape[1], self.n_latent).to(self.device)
        
        # Load Deep CCA if it was used during training
        checkpoint = torch.load(os.path.join(self.model_path, "ckpt.pth"))
        self.E_A.load_state_dict(checkpoint['E_A'])
        self.E_B.load_state_dict(checkpoint['E_B'])
        self.G_A.load_state_dict(checkpoint['G_A'])
        self.G_B.load_state_dict(checkpoint['G_B'])
        
        if self.use_deep_cca and 'deep_cca' in checkpoint:
            self.deep_cca = DeepCCA(
                latent_dim=self.n_latent,
                cca_dim=self.cca_dim,
                hidden_dims=self.cca_hidden_dims
            ).to(self.device)
            self.deep_cca.load_state_dict(checkpoint['deep_cca'])
            print("Deep CCA loaded from checkpoint")

        x_A = torch.from_numpy(self.emb_A).float().to(self.device)
        x_B = torch.from_numpy(self.emb_B).float().to(self.device)

        # For evaluation, we don't use adjacency matrices (full batch inference)
        z_A = self.E_A(x_A)
        z_B = self.E_B(x_B)
        
        # Apply Deep CCA alignment during evaluation if available
        if self.use_deep_cca and hasattr(self, 'deep_cca'):
            z_A_aligned, z_B_aligned, _ = self._apply_deep_cca_alignment(z_A, z_B)
            z_A = z_A_aligned
            z_B = z_B_aligned

        x_AtoB = self.G_B(z_A)
        x_BtoA = self.G_A(z_B)

        end_time = time.time()
        
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate((z_A.detach().cpu().numpy(), z_B.detach().cpu().numpy()), axis=0)
        self.data_Aspace = np.concatenate((self.emb_A, x_BtoA.detach().cpu().numpy()), axis=0)
        self.data_Bspace = np.concatenate((x_AtoB.detach().cpu().numpy(), self.emb_B), axis=0)

    def get_imputed_df(self, 
                       scale = 'scaled' # if scale=='log', then restore expression after log1p
                       ):

        x_BtoA = self.data_Aspace[self.emb_A.shape[0]:]
        x_AtoB = self.data_Bspace[:self.emb_A.shape[0]]
        if scale == 'log':
            x_BtoA = x_BtoA * self.adata_A.var['std'].values.reshape(1, -1) + self.adata_A.var['mean'].values.reshape(1, -1)
            x_AtoB = x_AtoB * self.adata_B.var['std'].values.reshape(1, -1) + self.adata_B.var['mean'].values.reshape(1, -1)
        imputed_df_BtoA = pd.DataFrame(x_BtoA, index=self.adata_B.obs.index, columns=self.adata_A.var.feature_name)
        imputed_df_BtoA = imputed_df_BtoA.groupby(imputed_df_BtoA.columns, axis=1).mean()
        imputed_df_AtoB = pd.DataFrame(x_AtoB, index=self.adata_A.obs.index, columns=self.adata_B.var.feature_name)
        imputed_df_AtoB = imputed_df_AtoB.groupby(imputed_df_AtoB.columns, axis=1).mean()
        self.imputed_df_BtoA = imputed_df_BtoA
        self.imputed_df_AtoB = imputed_df_AtoB

    def integrate_datasets_links(self, # Use this function for N >= 3 datasets when provided features links for MNN
                                 input_feats,
                                 feat_links_MNN, # A list of index pairs for feature linkages between features in "inputs_MNN"
                                 input_MNN=None, # A list of features matrices for finding MNN pairs between datasets; set as the same as input_feats if "input_MNN=None"
                                 ):
        begin_time = time.time()
        print("Beginning time: ", time.asctime(time.localtime(begin_time)))
        print(f"Using Harmony for MNN: {self.use_harmony}")
        print(f"Using Deep CCA: {self.use_deep_cca}")
        print(f"Using GAT: {self.use_gat}")
        
        num_datasets = len(input_feats)
        assert len(feat_links_MNN) == (num_datasets-1)
        self.E_dict = {}
        self.G_dict = {}
        params_G = []
        for i in range(num_datasets):
            self.E_dict[i] = encoder(input_feats[i].shape[1], self.n_latent,
                                    use_gat=self.use_gat,
                                    gat_hidden=self.gat_hidden,
                                    gat_heads=self.gat_heads,
                                    dropout=self.gat_dropout).to(self.device)
            params_G += self.E_dict[i].parameters()
            self.G_dict[i] = generator(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.G_dict[i].parameters()
        
        # Initialize Deep CCA modules for each consecutive pair if enabled
        self.deep_cca_dict = {}
        if self.use_deep_cca:
            for i in range(num_datasets-1):
                self.deep_cca_dict[i] = DeepCCA(
                    latent_dim=self.n_latent,
                    cca_dim=self.cca_dim,
                    hidden_dims=self.cca_hidden_dims
                ).to(self.device)
                params_G += list(self.deep_cca_dict[i].parameters())
        
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)

        self.D_dict = {}
        params_D = []
        for i in range(num_datasets-1):
            self.D_dict[i] = discriminator(self.n_latent).to(self.device)
            params_D += self.D_dict[i].parameters()
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.001)

        for i in range(num_datasets):
            self.E_dict[i].train()
            self.G_dict[i].train()
        for i in range(num_datasets-1):
            self.D_dict[i].train()
        if self.use_deep_cca:
            for i in range(num_datasets-1):
                self.deep_cca_dict[i].train()

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            x_dict = {}
            z_dict = {}
            z_aligned_dict = {}  # For Deep CCA aligned representations
            K_dict = {}
            K_z_dict = {}
            if input_MNN != None:
                assert len(input_MNN) == num_datasets
                x_MNN_dict = {}
            for i in range(num_datasets):
                index_i = np.random.choice(np.arange(input_feats[i].shape[0]), size=self.batch_size)
                x_dict[i] = torch.from_numpy(input_feats[i][index_i, :]).float().to(self.device)
                if input_MNN != None:
                    assert input_MNN[i].shape[0] == input_feats[i].shape[0]
                    x_MNN_dict[i] = input_MNN[i][index_i, :]
                
                # Apply GAT if enabled
                if self.use_gat:
                    adj_i = self._get_gat_adjacency_batch(x_dict[i], k=self.gat_knn)
                    z_dict[i] = self.E_dict[i](x_dict[i], adj_i)
                else:
                    z_dict[i] = self.E_dict[i](x_dict[i])
                    
                K_dict[i] = torch.exp(-torch.mean((x_dict[i].view(self.batch_size, 1, -1) - x_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)
                K_z_dict[i] = torch.exp(-torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)

            # Apply Deep CCA alignment for each consecutive pair
            loss_CCA_total = torch.tensor(0.0).to(self.device)
            if self.use_deep_cca:
                for i in range(num_datasets-1):
                    z_i_aligned, z_i1_aligned, loss_CCA = self._apply_deep_cca_alignment_multi(
                        z_dict[i], z_dict[i+1], self.deep_cca_dict[i]
                    )
                    # Store aligned representations
                    if i == 0:
                        z_aligned_dict[i] = z_i_aligned
                    z_aligned_dict[i+1] = z_i1_aligned
                    loss_CCA_total += loss_CCA
            else:
                # If no Deep CCA, use original representations
                for i in range(num_datasets):
                    z_aligned_dict[i] = z_dict[i]

            # discriminator loss:
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = 0
                for i in range(num_datasets-1):
                    # Use aligned representations for discriminator
                    z_i = z_aligned_dict[i]
                    z_i1 = z_aligned_dict[i+1]
                    loss_D += (torch.log(1 + torch.exp(-self.D_dict[i](z_i))) + torch.log(1 + torch.exp(self.D_dict[i](z_i1)))).mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE = 0
            for i in range(num_datasets):
                loss_AE += torch.mean((self.G_dict[i](z_aligned_dict[i]) - x_dict[i])**2)

            # latent align loss:
            loss_LA = 0
            for i in range(num_datasets-1):
                loss_LA += torch.mean((z_aligned_dict[i] - self.E_dict[i+1](self.G_dict[i+1](z_aligned_dict[i])))**2)
                loss_LA += torch.mean((z_aligned_dict[i+1] - self.E_dict[i](self.G_dict[i](z_aligned_dict[i+1])))**2)

            # generator loss
            loss_G_GAN = 0
            for i in range(num_datasets-1):
                z_i = z_aligned_dict[i]
                z_i1 = z_aligned_dict[i+1]
                loss_G_GAN += -(torch.log(1 + torch.exp(-self.D_dict[i](z_i))) + torch.log(1 + torch.exp(self.D_dict[i](z_i1)))).mean()

            # geometric structure loss
            loss_Geo = 0
            for i in range(num_datasets):
                loss_Geo += - torch.clamp(cos(K_dict[i], K_z_dict[i]), max=0.975).mean()

            # MNN loss - UPDATED WITH HARMONY
            loss_MNN = 0
            for i in range(num_datasets-1):
                if input_MNN != None:
                    # Create batch labels for this pair
                    batch_labels_i = np.zeros(self.batch_size)
                    batch_labels_i1 = np.ones(self.batch_size)
                    
                    # Use harmonized MNN pairs
                    Sim = self._get_harmonized_mnn_pairs(
                        x_MNN_dict[i][:, feat_links_MNN[i][0]], 
                        x_MNN_dict[i+1][:, feat_links_MNN[i][1]],
                        batch_labels_i,
                        batch_labels_i1
                    )
                else:
                    # Create batch labels for this pair
                    batch_labels_i = np.zeros(self.batch_size)
                    batch_labels_i1 = np.ones(self.batch_size)
                    
                    # Use harmonized MNN pairs
                    Sim = self._get_harmonized_mnn_pairs(
                        x_dict[i][:, feat_links_MNN[i][0]], 
                        x_dict[i+1][:, feat_links_MNN[i][1]],
                        batch_labels_i,
                        batch_labels_i1
                    )
                Sim = torch.from_numpy(Sim).float().to(self.device)
                # Use aligned representations for MNN loss
                z_dist = torch.mean((z_aligned_dict[i].view(self.batch_size, 1, -1) - z_aligned_dict[i+1].view(1, self.batch_size, -1))**2, dim=2)
                loss_MNN += torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = (self.lambdaGAN * loss_G_GAN + 
                     self.lambdaAE * loss_AE + 
                     self.lambdaLA * loss_LA + 
                     self.lambdaMNN * loss_MNN + 
                     self.lambdaGeo * loss_Geo +
                     self.lambdaCCA * loss_CCA_total)  # Add CCA loss
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 200:
                harmony_status = " (Harmony)" if self.use_harmony else ""
                cca_status = " (Deep CCA)" if self.use_deep_cca else ""
                gat_status = " (GAT)" if self.use_gat else ""
                print(f"step {step}{harmony_status}{cca_status}{gat_status}, loss_D={loss_D:.6f}, loss_GAN={loss_G_GAN:.6f}, loss_AE={self.lambdaAE*loss_AE:.6f}, loss_Geo={self.lambdaGeo*loss_Geo:.6f}, loss_LA={self.lambdaLA*loss_LA:.6f}, loss_MNN={self.lambdaMNN*loss_MNN:.6f}, loss_CCA={self.lambdaCCA*loss_CCA_total:.6f}")

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        begin_time = time.time()
        print("Beginning time: ", time.asctime(time.localtime(begin_time)))

        for i in range(num_datasets):
            self.E_dict[i].train()
            z_dict[i] = self.E_dict[i](torch.from_numpy(input_feats[i]).float().to(self.device))

        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate([z_dict[i].detach().cpu().numpy() for i in range(num_datasets)], axis=0)

    def _apply_deep_cca_alignment_multi(self, z_A, z_B, deep_cca_module):
        """Apply Deep CCA alignment for multi-dataset case"""
        if not self.use_deep_cca or deep_cca_module is None:
            return z_A, z_B, torch.tensor(0.0).to(self.device)
        
        try:
            # Apply Deep CCA projection
            projected_A, projected_B = deep_cca_module(z_A, z_B)
            
            # Compute CCA loss
            cca_loss = deep_cca_loss(projected_A, projected_B)
            
            # Compute CCA-aligned representations using traditional CCA
            z_A_aligned, z_B_aligned = compute_cca_alignment(projected_A, projected_B, cca_dim=self.cca_dim)
            
            # Mix original and CCA-aligned representations
            mixing_ratio = self.cca_mixing_ratio
            z_A_mixed = (1 - mixing_ratio) * z_A + mixing_ratio * z_A_aligned
            z_B_mixed = (1 - mixing_ratio) * z_B + mixing_ratio * z_B_aligned
            
            return z_A_mixed, z_B_mixed, cca_loss
            
        except Exception as e:
            print(f"Deep CCA alignment failed: {e}")
            return z_A, z_B, torch.tensor(0.0).to(self.device)

    def integrate_datasets_feats(self, # Use this function for N >= 3 datasets when provided linked features for MNN
                                 input_feats,
                                 paired_input_MNN, # In the form of [[link_feat_data1, link_feat_data2], ..., [link_feat_data(N_1), link_feat_dataN]]
                                 ):
        begin_time = time.time()
        print("Beginning time: ", time.asctime(time.localtime(begin_time)))
        print(f"Using Harmony for MNN: {self.use_harmony}")
        print(f"Using Deep CCA: {self.use_deep_cca}")
        print(f"Using GAT: {self.use_gat}")
        
        num_datasets = len(input_feats)
        self.E_dict = {}
        self.G_dict = {}
        params_G = []
        for i in range(num_datasets):
            self.E_dict[i] = encoder(input_feats[i].shape[1], self.n_latent,
                                    use_gat=self.use_gat,
                                    gat_hidden=self.gat_hidden,
                                    gat_heads=self.gat_heads,
                                    dropout=self.gat_dropout).to(self.device)
            params_G += self.E_dict[i].parameters()
            self.G_dict[i] = generator(input_feats[i].shape[1], self.n_latent).to(self.device)
            params_G += self.G_dict[i].parameters()
        
        # Initialize Deep CCA modules for each consecutive pair if enabled
        self.deep_cca_dict = {}
        if self.use_deep_cca:
            for i in range(num_datasets-1):
                self.deep_cca_dict[i] = DeepCCA(
                    latent_dim=self.n_latent,
                    cca_dim=self.cca_dim,
                    hidden_dims=self.cca_hidden_dims
                ).to(self.device)
                params_G += list(self.deep_cca_dict[i].parameters())
        
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)

        self.D_dict = {}
        params_D = []
        for i in range(num_datasets-1):
            self.D_dict[i] = discriminator(self.n_latent).to(self.device)
            params_D += self.D_dict[i].parameters()
        optimizer_D = optim.Adam(params_D, lr=0.001, weight_decay=0.001)

        for i in range(num_datasets):
            self.E_dict[i].train()
            self.G_dict[i].train()
        for i in range(num_datasets-1):
            self.D_dict[i].train()
        if self.use_deep_cca:
            for i in range(num_datasets-1):
                self.deep_cca_dict[i].train()

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            x_dict = {}
            z_dict = {}
            z_aligned_dict = {}  # For Deep CCA aligned representations
            K_dict = {}
            K_z_dict = {}
            assert len(paired_input_MNN) == (num_datasets - 1)
            x_MNN_dict_0 = {}
            x_MNN_dict_1 = {}
            for i in range(num_datasets):
                index_i = np.random.choice(np.arange(input_feats[i].shape[0]), size=self.batch_size)
                x_dict[i] = torch.from_numpy(input_feats[i][index_i, :]).float().to(self.device)
                if i < (num_datasets-1):
                    x_MNN_dict_0[i] = paired_input_MNN[i][0][index_i, :]
                if i > 0:
                    x_MNN_dict_1[i-1] = paired_input_MNN[i-1][1][index_i, :]
                
                # Apply GAT if enabled
                if self.use_gat:
                    adj_i = self._get_gat_adjacency_batch(x_dict[i], k=self.gat_knn)
                    z_dict[i] = self.E_dict[i](x_dict[i], adj_i)
                else:
                    z_dict[i] = self.E_dict[i](x_dict[i])
                    
                K_dict[i] = torch.exp(-torch.mean((x_dict[i].view(self.batch_size, 1, -1) - x_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)
                K_z_dict[i] = torch.exp(-torch.mean((z_dict[i].view(self.batch_size, 1, -1) - z_dict[i].view(1, self.batch_size, -1))**2, dim=2)/2)

            # Apply Deep CCA alignment for each consecutive pair
            loss_CCA_total = torch.tensor(0.0).to(self.device)
            if self.use_deep_cca:
                for i in range(num_datasets-1):
                    z_i_aligned, z_i1_aligned, loss_CCA = self._apply_deep_cca_alignment_multi(
                        z_dict[i], z_dict[i+1], self.deep_cca_dict[i]
                    )
                    # Store aligned representations
                    if i == 0:
                        z_aligned_dict[i] = z_i_aligned
                    z_aligned_dict[i+1] = z_i1_aligned
                    loss_CCA_total += loss_CCA
            else:
                # If no Deep CCA, use original representations
                for i in range(num_datasets):
                    z_aligned_dict[i] = z_dict[i]

            # discriminator loss:
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = 0
                for i in range(num_datasets-1):
                    # Use aligned representations for discriminator
                    z_i = z_aligned_dict[i]
                    z_i1 = z_aligned_dict[i+1]
                    loss_D += (torch.log(1 + torch.exp(-self.D_dict[i](z_i))) + torch.log(1 + torch.exp(self.D_dict[i](z_i1)))).mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE = 0
            for i in range(num_datasets):
                loss_AE += torch.mean((self.G_dict[i](z_aligned_dict[i]) - x_dict[i])**2)

            # latent align loss:
            loss_LA = 0
            for i in range(num_datasets-1):
                loss_LA += torch.mean((z_aligned_dict[i] - self.E_dict[i+1](self.G_dict[i+1](z_aligned_dict[i])))**2)
                loss_LA += torch.mean((z_aligned_dict[i+1] - self.E_dict[i](self.G_dict[i](z_aligned_dict[i+1])))**2)

            # generator loss
            loss_G_GAN = 0
            for i in range(num_datasets-1):
                z_i = z_aligned_dict[i]
                z_i1 = z_aligned_dict[i+1]
                loss_G_GAN += -(torch.log(1 + torch.exp(-self.D_dict[i](z_i))) + torch.log(1 + torch.exp(self.D_dict[i](z_i1)))).mean()

            # geometric structure loss
            loss_Geo = 0
            for i in range(num_datasets):
                loss_Geo += - torch.clamp(cos(K_dict[i], K_z_dict[i]), max=0.975).mean()

            # MNN loss - UPDATED WITH HARMONY
            loss_MNN = 0
            for i in range(num_datasets-1):
                # Create batch labels for this pair
                batch_labels_0 = np.zeros(self.batch_size)
                batch_labels_1 = np.ones(self.batch_size)
                
                # Use harmonized MNN pairs
                Sim = self._get_harmonized_mnn_pairs(
                    x_MNN_dict_0[i], 
                    x_MNN_dict_1[i], 
                    batch_labels_0,
                    batch_labels_1
                )
                Sim = torch.from_numpy(Sim).float().to(self.device)
                # Use aligned representations for MNN loss
                z_dist = torch.mean((z_aligned_dict[i].view(self.batch_size, 1, -1) - z_aligned_dict[i+1].view(1, self.batch_size, -1))**2, dim=2)
                loss_MNN += torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = (self.lambdaGAN * loss_G_GAN + 
                     self.lambdaAE * loss_AE + 
                     self.lambdaLA * loss_LA + 
                     self.lambdaMNN * loss_MNN + 
                     self.lambdaGeo * loss_Geo +
                     self.lambdaCCA * loss_CCA_total)  # Add CCA loss
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 2000:
                harmony_status = " (Harmony)" if self.use_harmony else ""
                cca_status = " (Deep CCA)" if self.use_deep_cca else ""
                gat_status = " (GAT)" if self.use_gat else ""
                print(f"step {step}{harmony_status}{cca_status}{gat_status}, loss_D={loss_D:.6f}, loss_GAN={loss_G_GAN:.6f}, loss_AE={self.lambdaAE*loss_AE:.6f}, loss_Geo={self.lambdaGeo*loss_Geo:.6f}, loss_LA={self.lambdaLA*loss_LA:.6f}, loss_MNN={self.lambdaMNN*loss_MNN:.6f}, loss_CCA={self.lambdaCCA*loss_CCA_total:.6f}")

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        begin_time = time.time()
        print("Beginning time: ", time.asctime(time.localtime(begin_time)))

        for i in range(num_datasets):
            self.E_dict[i].train()
            z_dict[i] = self.E_dict[i](torch.from_numpy(input_feats[i]).float().to(self.device))

        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.eval_time = end_time - begin_time
        print("Evaluating takes %.2f seconds" % self.eval_time)

        self.latent = np.concatenate([z_dict[i].detach().cpu().numpy() for i in range(num_datasets)], axis=0)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {}
        for i in range(num_datasets):
            state['E_%d' % i] = self.E_dict[i].state_dict()
            state['G_%d' % i] = self.G_dict[i].state_dict()
        
        # Save Deep CCA states if used
        if self.use_deep_cca:
            for i in range(num_datasets-1):
                state['deep_cca_%d' % i] = self.deep_cca_dict[i].state_dict()

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))
