import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import harmonypy as hm

from scmodal.networks import *
from scmodal.utils import *

class Model(object):
    def __init__(self, batch_size=500, training_steps=10000, seed=1234, n_latent=20,
                 lambdaAE = 10.0, lambdaLA = 10.0, lambdaMNN = 1.0, lambdaGeo = 10.0, lambdaGAN = 1.0, n_KNN = 30,
                 model_path="models", data_path="data", result_path="results", 
                 use_harmony=True, harmony_max_iter_harmony=20, harmony_sigma=0.1, harmony_theta=2.0):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
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

    def _harmonize_embeddings(self, embeddings, batch_labels):
        """Apply Harmony batch correction to embeddings"""
        if not self.use_harmony:
            return embeddings
            
        try:
            # Create metadata DataFrame for Harmony
            meta_data = pd.DataFrame({
                'batch': batch_labels
            })
            
            print(f"Running Harmony on embeddings shape: {embeddings.shape}, batches: {np.unique(batch_labels)}")
            
            # Run Harmony with proper parameters
            ho = hm.run_harmony(
                embeddings, 
                meta_data, 
                vars_use=['batch'],  # CRITICAL: This was missing!
                max_iter_harmony=self.harmony_max_iter_harmony,
                sigma=self.harmony_sigma,
                theta=self.harmony_theta,
                verbose=False
            )
            harmonized_embeddings = ho.Z_corr.T
            print(f"Harmony completed successfully. Output shape: {harmonized_embeddings.shape}")
            return harmonized_embeddings
            
        except Exception as e:
            print(f"Warning: Harmony failed with error {e}. Using original embeddings.")
            return embeddings

    def _get_harmonized_mnn_pairs(self, feat_A, feat_B, batch_labels_A, batch_labels_B):
        """Get MNN pairs using harmonized embeddings"""
        if not self.use_harmony:
            return acquire_pairs(feat_A, feat_B, k=self.n_KNN)
            
        try:
            # Combine features and batch labels
            combined_feats = np.vstack([feat_A, feat_B])
            combined_batches = np.concatenate([batch_labels_A, batch_labels_B])
            
            print(f"Combined features shape: {combined_feats.shape}, batches: {np.unique(combined_batches)}")
            
            # Apply Harmony
            harmonized_feats = self._harmonize_embeddings(combined_feats, combined_batches)
            
            # Split back into A and B
            harmonized_A = harmonized_feats[:len(feat_A)]
            harmonized_B = harmonized_feats[len(feat_A):]
            
            print(f"Harmonized A shape: {harmonized_A.shape}, B shape: {harmonized_B.shape}")
            
            # Get MNN pairs on harmonized embeddings
            Sim = acquire_pairs(harmonized_A, harmonized_B, k=self.n_KNN)
            print(f"MNN pairs found: {np.sum(Sim)}")
            return Sim
            
        except Exception as e:
            print(f"Warning: Harmonized MNN failed: {e}. Using regular MNN.")
            return acquire_pairs(feat_A, feat_B, k=self.n_KNN)

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
        self.batch_labels_A = np.zeros(self.emb_A.shape[0])  # Batch 0 for dataset A
        self.batch_labels_B = np.ones(self.emb_B.shape[0])   # Batch 1 for dataset B

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
        self.batch_labels_A = np.zeros(self.emb_A.shape[0])  # Batch 0 for dataset A
        self.batch_labels_B = np.ones(self.emb_B.shape[0])   # Batch 1 for dataset B
        
        if layer_adata_A_MNN is None:
            self.feat_A_MNN = self.emb_A
        else:
            self.feat_A_MNN = adata_A.obsm[layer_adata_A_MNN]
        if layer_adata_B_MNN is None:
            self.feat_B_MNN = self.emb_B
        else:
            self.feat_B_MNN = adata_B.obsm[layer_adata_B_MNN]

    # ... rest of your train, eval, and other methods remain the same ...
    # Just make sure to use the corrected _get_harmonized_mnn_pairs method in your training loops

    def train(self):
        begin_time = time.time()
        print("Beginning time: ", time.asctime(time.localtime(begin_time)))
        print(f"Using Harmony for MNN: {self.use_harmony}")
        
        self.E_A = encoder(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.E_B = encoder(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.G_A = generator(self.emb_A.shape[1], self.n_latent).to(self.device)
        self.G_B = generator(self.emb_B.shape[1], self.n_latent).to(self.device)
        self.D_Z = discriminator(self.n_latent).to(self.device)
        params_G = list(self.E_A.parameters()) + list(self.E_B.parameters()) + list(self.G_A.parameters()) + list(self.G_B.parameters())
        optimizer_G = optim.Adam(params_G, lr=0.001, weight_decay=0.001)
        optimizer_D = optim.Adam(list(self.D_Z.parameters()), lr=0.001, weight_decay=0.001)
        self.E_A.train()
        self.E_B.train()
        self.G_A.train()
        self.G_B.train()
        self.D_Z.train()

        N_A = self.emb_A.shape[0]
        N_B = self.emb_B.shape[0]

        for step in range(self.training_steps):
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            index_A = np.random.choice(np.arange(N_A), size=self.batch_size)
            index_B = np.random.choice(np.arange(N_B), size=self.batch_size)
            x_A = torch.from_numpy(self.emb_A[index_A, :]).float().to(self.device)
            x_B = torch.from_numpy(self.emb_B[index_B, :]).float().to(self.device)
            z_A = self.E_A(x_A)
            z_B = self.E_B(x_B)
            x_AtoB = self.G_B(z_A)
            x_BtoA = self.G_A(z_B)
            x_Arecon = self.G_A(z_A)
            x_Brecon = self.G_B(z_B)
            z_AtoB = self.E_B(x_AtoB)
            z_BtoA = self.E_A(x_BtoA)
            K_A = torch.mean((x_A.view(self.batch_size, 1, -1) - x_A.view(1, self.batch_size, -1))**2, dim=2)
            K_A = torch.exp(-K_A/2)
            K_B_z = torch.mean((z_B.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
            K_B_z = torch.exp(-K_B_z/2)
            K_B = torch.mean((x_B.view(self.batch_size, 1, -1) - x_B.view(1, self.batch_size, -1))**2, dim=2)
            K_B = torch.exp(-K_B/2)
            K_A_z = torch.mean((z_A.view(self.batch_size, 1, -1) - z_A.view(1, self.batch_size, -1))**2, dim=2)
            K_A_z = torch.exp(-K_A_z/2)

            # discriminator loss:
            for _ in range(5):
                optimizer_D.zero_grad()
                loss_D = (torch.log(1 + torch.exp(-self.D_Z(z_A))) + torch.log(1 + torch.exp(self.D_Z(z_B)))).mean()
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # autoencoder loss:
            loss_AE_A = torch.mean((x_Arecon - x_A)**2)
            loss_AE_B = torch.mean((x_Brecon - x_B)**2)
            loss_AE = loss_AE_A + loss_AE_B

            # latent align loss:
            loss_LA_AtoB = torch.mean((z_A - z_AtoB)**2)
            loss_LA_BtoA = torch.mean((z_B - z_BtoA)**2)
            loss_LA = loss_LA_AtoB + loss_LA_BtoA

            # generator loss
            loss_G_GAN = -(torch.log(1 + torch.exp(-self.D_Z(z_A))) + torch.log(1 + torch.exp(self.D_Z(z_B)))).mean()

            # geometric structure loss
            loss_Geo = - (torch.clamp(cos(K_A, K_A_z), max=0.975).mean() + torch.clamp(cos(K_B, K_B_z), max=0.975).mean())

            # MNN loss - UPDATED WITH HARMONY
            if hasattr(self, 'feat_A_MNN') and hasattr(self, 'feat_B_MNN'):
                # Use additional features for MNN if available
                Sim = self._get_harmonized_mnn_pairs(
                    self.feat_A_MNN[index_A], 
                    self.feat_B_MNN[index_B],
                    self.batch_labels_A[index_A],
                    self.batch_labels_B[index_B]
                )
            else:
                # Use shared genes for MNN
                Sim = self._get_harmonized_mnn_pairs(
                    self.emb_A[index_A, :self.shared_gene_num], 
                    self.emb_B[index_B, :self.shared_gene_num],
                    self.batch_labels_A[index_A],
                    self.batch_labels_B[index_B]
                )
            
            Sim = torch.from_numpy(Sim).float().to(self.device)
            z_dist = torch.mean((z_A.view(self.batch_size, 1, -1) - z_B.view(1, self.batch_size, -1))**2, dim=2)
            loss_MNN = torch.sum(Sim * z_dist) / torch.sum(Sim)

            optimizer_G.zero_grad()
            loss_G = self.lambdaGAN * loss_G_GAN + self.lambdaAE * loss_AE + self.lambdaLA * loss_LA + self.lambdaMNN * loss_MNN + self.lambdaGeo*loss_Geo
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(params_G, 5.0)
            optimizer_G.step()

            if not step % 2000:
                harmony_status = " (Harmony)" if self.use_harmony else ""
                print(f"step {step}{harmony_status}, loss_D={loss_D:.6f}, loss_GAN={loss_G_GAN:.6f}, loss_AE={self.lambdaAE*loss_AE:.6f}, loss_Geo={self.lambdaGeo*loss_Geo:.6f}, loss_LA={self.lambdaLA*loss_LA:.6f}, loss_MNN={self.lambdaMNN*loss_MNN:.6f}")

        end_time = time.time()
        print("Ending time: ", time.asctime(time.localtime(end_time)))
        self.train_time = end_time - begin_time
        print("Training takes %.2f seconds" % self.train_time)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'E_A': self.E_A.state_dict(), 'E_B': self.E_B.state_dict(),
                 'G_A': self.G_A.state_dict(), 'G_B': self.G_B.state_dict()}

        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))
