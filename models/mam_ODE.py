import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
from einops import rearrange
from torchdiffeq import odeint
import numpy as np


class PatchEmbedding(nn.Module):
    """
    Convert time series to patch embeddings
    Input: (B, seq_len, channels)
    Output: (B, num_patches, patch_dim)
    """
    def __init__(self, seq_len, patch_len, stride, channels, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.channels = channels
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_len) // stride + 1
        
        # Patch embedding: flatten patch + linear projection
        self.patch_embedding = nn.Linear(patch_len * channels, d_model)
        
    def forward(self, x):
        """
        x: (B, seq_len, channels)
        Returns: (B, num_patches, d_model)
        """
        B, L, C = x.shape
        
        # Extract patches using unfold
        # (B, C, L) -> (B, C, num_patches, patch_len)
        x = x.transpose(1, 2)  # (B, C, L)
        patches = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # Reshape: (B, C, num_patches, patch_len) -> (B, num_patches, C*patch_len)
        patches = rearrange(patches, 'b c n p -> b n (c p)')
        
        # Embed patches
        patch_embed = self.patch_embedding(patches)  # (B, num_patches, d_model)
        
        return patch_embed


class AODEFunc(nn.Module):
    """ODE dynamics for Koopman observables"""
    def __init__(self, N: int, hidden_dim: int = 64, prune_hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prune_hidden, hidden_dim), 
            nn.SiLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, prune_hidden), 
            nn.Tanh(),
        )
        self.lat2out = nn.Sequential(
            nn.Linear(prune_hidden, hidden_dim), 
            nn.SiLU(), 
            nn.Linear(hidden_dim, N), 
            nn.Tanh()
        )

    def forward(self, t: torch.Tensor, a: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class _JointODEFunc(nn.Module):
    """Augmented dynamics for patch-level Koopman observables"""
    def __init__(self, model: 'PatchKoopmanODE', context: torch.Tensor):
        super().__init__()
        self.model = model
        self.context = context

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        M = self.model
        N = M.prune_hidden
        
        if t.dim() == 0:
            t = t.to(s.device)
        
        a = s[:, :N]      # (B, prune_hidden) - compressed Koopman observables
        z = s[:, N:]      # (B, d_model) - patch representation
        
        # Koopman dynamics
        da = M.a_ode(t, a, self.context)
        
        # Get circulant matrix first row
        c = M.a_ode.lat2out(a)  # (B, d_model) - first row of circulant matrix
        
        # Construct circulant matrix from c
        B, D = c.shape
        circulant_matrix = torch.zeros(B, D, D, device=c.device, dtype=c.dtype)
        
        # Build circulant matrix: each row is cyclic shift of first row
        for i in range(D):
            circulant_matrix[:, i, :] = torch.roll(c, shifts=i, dims=1)
        
        # Matrix-vector multiplication: C @ z
        dz = torch.bmm(circulant_matrix, z.unsqueeze(-1)).squeeze(-1)  # (B, d_model)
        
        return torch.cat([da, dz], dim=-1)


class PatchKoopmanODE(nn.Module):
    """
    Patch-based Koopman ODE Predictor
    Operates on patch-level representations
    """
    def __init__(self,
                 seq_len=96,
                 pred_len=96,
                 patch_len=16,
                 stride=8,
                 channels=8,
                 d_model=64,
                 hidden_dim=64,
                 prune_hidden=16):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.channels = channels
        self.d_model = d_model
        self.prune_hidden = prune_hidden
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            seq_len=seq_len,
            patch_len=patch_len,
            stride=stride,
            channels=channels,
            d_model=d_model
        )
        self.num_patches = self.patch_embed.num_patches
        
        # Calculate target number of patches for prediction
        self.target_patches = (pred_len - 1) // stride + 1
        
        # Temporal convolution over patches
        self.patch_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=2,
            groups=d_model
        )
        
        # Compress patch sequence to initial Koopman observable
        self.compress = nn.Sequential(
            nn.Linear(d_model * self.num_patches, prune_hidden),
            nn.Dropout(0.1)
        )
        
        # Koopman ODE
        self.a_ode = AODEFunc(d_model, hidden_dim, prune_hidden)
        self.span_A = nn.Parameter(torch.tensor(0.5))
        
        # Patch decoder: convert patch embedding back to time series
        # self.patch_decoder = nn.Linear(d_model, patch_len * channels)

        self.patch_decoder = nn.Sequential(nn.Linear(d_model, patch_len * channels),
                                           nn.Dropout(0.1),
                                           nn.SiLU(),
                                           nn.Linear(patch_len * channels, patch_len * channels))
        
    def forward(self, x):
        """
        x: (B, seq_len, channels)
        Returns:
            x_rec: (B, seq_len, channels) reconstructed
            x_pred: (B, pred_len, channels) predicted
        """
        B, T, C = x.shape
        device = x.device
        
        # 1) Patch embedding
        patch_embed = self.patch_embed(x)  # (B, num_patches, d_model)
        
        # 2) Temporal convolution over patches
        patch_feat = rearrange(patch_embed, 'b n d -> b d n')
        patch_feat = F.silu(self.patch_conv(patch_feat))[:, :, :self.num_patches]
        patch_feat = rearrange(patch_feat, 'b d n -> b n d')
        
        # 3) Initialize Koopman observable
        patch_flat = patch_feat.reshape(B, -1)
        a0 = self.compress(patch_flat)  # (B, prune_hidden)
        z0 = patch_embed[:, -1, :]  # Last patch as initial state (B, d_model)
        
        # 4) ODE integration to evolve patches
        span_A  = torch.clamp(self.span_A, min=1e-8, max=5.0)
        tspan   = torch.linspace(0, float(span_A * self.target_patches), 
                              self.target_patches + 1,
                              device=device, dtype=x.dtype)
        
        s0 = torch.cat([a0, z0], dim=-1)
        joint_func = _JointODEFunc(self, patch_flat)
        
        sol = odeint(joint_func, s0, tspan, method="euler", rtol=1e-3, atol=1e-4)
        # (target_patches+1, B, prune_hidden+d_model)
        
        # 5) Decode predicted patches
        z_traj = sol[1:, :, -self.d_model:]  # (target_patches, B, d_model)
        
        # Convert patches back to time series
        pred_patches = []
        for k in range(self.target_patches):
            patch_vec = self.patch_decoder(z_traj[k])  # (B, patch_len*channels)
            patch_vec = patch_vec.reshape(B, self.patch_len, C)
            pred_patches.append(patch_vec)
        
        # Concatenate patches with overlap handling
        x_pred = self._reconstruct_from_patches(pred_patches)  # (B, pred_len, C)
        
        # Reconstruct input (simple passthrough for now)
        x_rec = x
        
        return x_rec, x_pred
    
    def _reconstruct_from_patches(self, patches):
        """
        Reconstruct time series from overlapping patches
        patches: list of (B, patch_len, C) tensors
        Returns: (B, pred_len, C)
        """
        B = patches[0].shape[0]
        C = patches[0].shape[2]
        device = patches[0].device
        
        # Simple reconstruction: use stride to place patches
        output = torch.zeros(B, self.pred_len, C, device=device)
        counts = torch.zeros(B, self.pred_len, C, device=device)
        
        for i, patch in enumerate(patches):
            start = i * self.stride
            end = min(start + self.patch_len, self.pred_len)
            patch_end = end - start
            
            output[:, start:end, :] += patch[:, :patch_end, :]
            counts[:, start:end, :] += 1
        
        # Average overlapping regions
        output = output / (counts + 1e-6)
        
        return output


class Model(nn.Module):
    """
    Patch-based Koopman ODE Forecasting Model
    """
    def __init__(self, configs, 
                 patch_len=48, 
                 stride=8,
                 d_model=64, 
                 hidden_dim=64):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        
        self.patch_len = (self.seq_len-self.pred_len)//2
        self.stride = stride
        
        # Patch-based Koopman ODE predictor
        self.predictor = PatchKoopmanODE(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            patch_len=patch_len,
            stride=stride,
            channels=self.channels,
            d_model=d_model,
            hidden_dim=hidden_dim
        )
        
    def forecast(self, x_enc):
        """
        Forecast with series stationarization
        """
        # Stationarization
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc
        
        # Patch-based Koopman forecasting
        x_rec, x_pred = self.predictor(x_enc)
        
        # De-stationarization
        forecast = x_pred * std_enc + mean_enc
        
        return forecast
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, :self.pred_len, :]
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented")