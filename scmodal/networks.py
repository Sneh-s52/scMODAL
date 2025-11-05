import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GraphAttentionLayer(nn.Module):
    """Simple Graph Attention Layer"""
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj=None):
        """
        Args:
            h: input features [batch_size, n_features]
            adj: adjacency matrix [batch_size, batch_size]. If None, use self-attention
        """
        batch_size = h.size(0)
        
        # Linear transformation
        Wh = torch.mm(h, self.W)  # [batch_size, out_features]
        
        # Self-attention mechanism
        if adj is None:
            # Compute attention coefficients using self-attention
            a_input = torch.cat([Wh.repeat(1, batch_size).view(batch_size * batch_size, -1),
                               Wh.repeat(batch_size, 1)], dim=1)
            a_input = a_input.view(batch_size, batch_size, 2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            
            # Create attention mask (attend to all nodes)
            attention = F.softmax(e, dim=1)
        else:
            # Use provided adjacency matrix
            e = self.leakyrelu(torch.matmul(Wh, self.a[:self.out_features, :]) + 
                             torch.matmul(Wh, self.a[self.out_features:, :]).t())
            attention = F.softmax(e * adj, dim=1)
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)


class GATEncoder(nn.Module):
    """Encoder with Graph Attention"""
    def __init__(self, n_input, n_latent, gat_hidden=256, gat_heads=1, dropout=0.1):
        super(GATEncoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.gat_hidden = gat_hidden
        
        # GAT layers
        self.attention_layers = nn.ModuleList()
        self.attention_layers.append(
            GraphAttentionLayer(n_input, gat_hidden, dropout=dropout)
        )
        self.attention_layers.append(
            GraphAttentionLayer(gat_hidden, n_latent, dropout=dropout)
        )
        
        # Optional: Multi-head attention
        self.multi_head_attentions = nn.ModuleList()
        for _ in range(gat_heads):
            head = nn.ModuleList([
                GraphAttentionLayer(n_input, gat_hidden, dropout=dropout),
                GraphAttentionLayer(gat_hidden, n_latent, dropout=dropout)
            ])
            self.multi_head_attentions.append(head)
        
        self.gat_heads = gat_heads
        self.dropout = dropout

    def forward(self, x, adj=None):
        """
        Args:
            x: input features [batch_size, n_input]
            adj: optional adjacency matrix [batch_size, batch_size]
        """
        if self.gat_heads > 1:
            # Multi-head attention
            head_outputs = []
            for head in self.multi_head_attentions:
                h = x
                for attn_layer in head:
                    h = attn_layer(h, adj)
                head_outputs.append(h)
            
            # Average all heads
            h = torch.mean(torch.stack(head_outputs), dim=0)
        else:
            # Single head attention
            h = x
            for attn_layer in self.attention_layers:
                h = attn_layer(h, adj)
        
        return h


class encoder(nn.Module):
    def __init__(self, n_input, n_latent, use_gat=False, gat_hidden=256, gat_heads=1, dropout=0.1):
        super(encoder, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        self.use_gat = use_gat
        n_hidden = 512

        if use_gat:
            # GAT-based encoder
            self.gat_encoder = GATEncoder(n_input, n_latent, gat_hidden, gat_heads, dropout)
        else:
            # Original MLP encoder
            self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
            self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))
            self.W_2 = nn.Parameter(torch.Tensor(self.n_latent, n_hidden).normal_(mean=0.0, std=0.1))
            self.b_2 = nn.Parameter(torch.Tensor(self.n_latent).normal_(mean=0.0, std=0.1))

    def forward(self, x, adj=None):
        if self.use_gat:
            z = self.gat_encoder(x, adj)
        else:
            h = F.relu(F.linear(x, self.W_1, self.b_1))
            z = F.linear(h, self.W_2, self.b_2)
        return z


class generator(nn.Module):
    def __init__(self, n_input, n_latent):
        super(generator, self).__init__()
        self.n_input = n_input
        self.n_latent = n_latent
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_latent).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(self.n_input, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(self.n_input).normal_(mean=0.0, std=0.1))

    def forward(self, z):
        h = F.relu(F.linear(z, self.W_1, self.b_1))
        x = F.linear(h, self.W_2, self.b_2)
        return x


class discriminator(nn.Module):
    def __init__(self, n_input):
        super(discriminator, self).__init__()
        self.n_input = n_input
        n_hidden = 512

        self.W_1 = nn.Parameter(torch.Tensor(n_hidden, self.n_input).normal_(mean=0.0, std=0.1))
        self.b_1 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_2 = nn.Parameter(torch.Tensor(n_hidden, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_2 = nn.Parameter(torch.Tensor(n_hidden).normal_(mean=0.0, std=0.1))

        self.W_3 = nn.Parameter(torch.Tensor(1, n_hidden).normal_(mean=0.0, std=0.1))
        self.b_3 = nn.Parameter(torch.Tensor(1).normal_(mean=0.0, std=0.1))

    def forward(self, x):
        h = F.relu(F.linear(x, self.W_1, self.b_1))
        h = F.relu(F.linear(h, self.W_2, self.b_2))
        score = F.linear(h, self.W_3, self.b_3)
        return torch.clamp(score, min=-50.0, max=50.0)
