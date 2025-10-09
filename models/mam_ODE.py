import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
from einops import rearrange
from torchdiffeq import odeint
import numpy as np
from data_provider.data_factory import data_provider


class FourierFilter(nn.Module):
    """
    Fourier Filter: to time-variant and time-invariant term
    """
    def __init__(self, mask_spectrum):
        super(FourierFilter, self).__init__()
        self.mask_spectrum = mask_spectrum
        
    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf*mask, dim=1)
        x_inv = x - x_var
        
        return x_var, x_inv


class AODEFunc(nn.Module):
    """d a / dt = f_a(t, a)  (a ∈ ℝ^N)"""
    def __init__(self, N: int, hidden_dim: int = 64, dim_C: int = 64, prune_hidden: int = 32, dim_Z: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prune_hidden, hidden_dim), 
            nn.SiLU(), 
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, prune_hidden), 
            nn.Tanh(),
        )
        self.lat2out = nn.Sequential(nn.Linear(prune_hidden, 8), nn.SiLU(), nn.Linear(8, N))
        self.lat2out_h = nn.Sequential(nn.Linear(prune_hidden, 8), nn.SiLU(), nn.Linear(8, N))
        self.lat2out_c = nn.Sequential(nn.Linear(prune_hidden, 8), nn.SiLU(), nn.Linear(8, dim_C))
        self.lat2out_z = nn.Sequential(nn.Linear(N, N), nn.SiLU(), nn.Linear(N, dim_Z))

    def forward(self, t: torch.Tensor, a: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class _JointODEFunc(nn.Module):
    """Augmented dynamics for **a, z** (removed b).

    Parameters
    ----------
    model : parent ODEPredictor (provides sub‑modules & hyper‑params)
    context : context vector for dynamics
    """
    def __init__(self, model: 'ODEPredictor', context: torch.Tensor):
        super().__init__()
        self.model = model
        self.context = context

    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:  # (B,*)
        M = self.model
        N = M.prune_hidden
        a = s[:, :N]                  # (B,N)
        z = s[:, N:]                  # (B,N)
        
        da = M.a_ode(t, a, self.context)
        d = M.a_ode.lat2out(a).unsqueeze(-1)          # (B,N,1)
        
        self.H = M.H
        H = M.H
        A_mat = torch.einsum('bij,bjk->bik', H.transpose(-1, -2), d * H)  # (B,N,N)
        
        dz = (A_mat @ z.unsqueeze(-1)).squeeze(-1)  # (B,N)
        
        return torch.cat([da, dz], dim=-1)


class Householder(nn.Module):
    """Householder reflection matrix constructor."""
    def __init__(self, N: int):
        super().__init__()
        self.N = N
    
    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B, N)
        device = v.device
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)
        I = torch.eye(self.N, device=device, dtype=v.dtype).unsqueeze(0)  # (1, N, N)
        H = I - 2 * torch.einsum('bi,bj->bij', v, v)  # (B, N, N)
        return H


class ODEPredictor(nn.Module):
    """
    ODE-based Koopman Predictor
    Utilize neural ODE to learn dynamics in latent space
    """
    def __init__(self,
                 enc_in=8,
                 input_len=96,
                 pred_len=96,
                 latent_dim=64,
                 hidden_dim=64,
                 prune_hidden=8,
                 d_conv=3):
        super(ODEPredictor, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.N = latent_dim
        self.prune_hidden = prune_hidden
        self.d_conv = d_conv
        
        # Learnable projections
        self.x_proj = nn.Sequential(
            nn.Linear(self.enc_in, hidden_dim), nn.SiLU(), nn.Dropout(0.1), 
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1), 
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1), 
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Dropout(0.1), 
            nn.Linear(hidden_dim, self.N)
        )
        self.x_proj_ = nn.Sequential(
            nn.Linear(self.N, hidden_dim), nn.SiLU(), 
            nn.Dropout(0.0), nn.Linear(hidden_dim, self.N)
        )

        # Conv over history
        self.conv = nn.Conv1d(
            in_channels=self.N,
            out_channels=self.N,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) * 2,
            dilation=2,
            groups=self.N
        )
        nn.init.kaiming_uniform_(self.conv.weight, mode="fan_in", nonlinearity="relu")

        # ODE nets & helpers
        self.span_A = nn.Parameter(torch.tensor(0.1))
        self.compress = nn.Sequential(
            nn.Linear(self.N * self.input_len, self.prune_hidden), 
            nn.Dropout(0.1)
        )
        dim_C = self.enc_in * self.N
        dim_Z = self.enc_in
        self.a_ode = AODEFunc(self.N, hidden_dim, dim_C, self.prune_hidden, dim_Z)
        self.householder = Householder(self.N)
        self.delta_raw = nn.Parameter(torch.zeros(self.pred_len))
        
        self.H = None

    def forward(self, x):
        """
        x: (B, L, C) input time series
        Returns:
            x_rec: (B, L, C) reconstructed input
            x_pred: (B, S, C) predicted future
        """
        B, T, _ = x.shape
        device = x.device

        # 1) History embedding → initial a0 & z0
        x_old = self.x_proj(x[:, :self.input_len])           # (B, L, N)
        hist = rearrange(x_old, "B L D -> B D L")            # (B, N, L)
        hist = F.silu(self.conv(hist))[:, :, -self.input_len:]   # (B, N, L)
        hist = rearrange(hist, "B D L -> B L D")
        hist_f = hist.reshape(B, -1)
        vector_F = self.compress(hist_f)
        a0 = vector_F
        z0 = self.x_proj(x[:, self.input_len - 1])          # (B, N)
        self.H = self.householder(self.x_proj_(z0))
        
        # 2) Time span setup - ensure on correct device
        span_A = torch.clamp(self.span_A, min=1e-8, max=7.0)
        tspan = torch.arange(1, self.pred_len + 1, device=device, dtype=torch.float32) * span_A
        tspan = torch.cat([torch.zeros(1, device=device, dtype=torch.float32), tspan], dim=0)
        
        # 3) ODE integration
        s0 = torch.cat([a0, z0], dim=-1)        # (B, prune_hidden+N)
        joint_func = _JointODEFunc(self, vector_F)
        sol = odeint(joint_func, s0, tspan, method="euler", rtol=1e-3, atol=1e-4)  # (L+1, B, D)

        # 4) Decode predictions
        z_traj = sol[1:, :, -self.N:]                      # (pred_len, B, N)
        c_traj = sol[1:, :, :self.prune_hidden]
        C = self.a_ode.lat2out_c(c_traj).permute(1, 0, 2).view(
            z_traj.shape[1], z_traj.shape[0], self.enc_in, self.N
        )
        
        # Generate predictions
        predictions = []
        for k in range(self.pred_len):
            y_hat_c = torch.einsum("bij,bj->bi", C[:, k, :], z_traj[k, :, :])
            Added_z = self.a_ode.lat2out_z(z_traj[k])
            y_hat = y_hat_c + Added_z
            predictions.append(y_hat)
        
        x_pred = torch.stack(predictions, dim=1)  # (B, pred_len, C)
        
        # Reconstruct input (simple pass-through for now)
        x_rec = x[:, :self.input_len, :]
        
        return x_rec, x_pred


class Model(nn.Module):
    """
    ODE-based Time Series Forecasting Model
    Adapted to Koopa framework structure
    """
    def __init__(self, configs, latent_dim=64, hidden_dim=64, num_blocks=1):
        """
        latent_dim: int, latent dimension of ODE embedding
        hidden_dim: int, hidden dimension of networks
        num_blocks: int, number of ODE blocks
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.enc_in = configs.enc_in
        self.input_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # ODE predictors
        self.ode_predictors = nn.ModuleList([
            ODEPredictor(
                enc_in=self.enc_in,
                input_len=self.input_len,
                pred_len=self.pred_len,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim
            )
            for _ in range(self.num_blocks)
        ])

    def forecast(self, x_enc):
        """
        Forecast future states with Series Stationarization.
        
        Parameters
        ----------
        x_enc : (B, T, S) input time series
        
        Returns
        -------
        forecast : (B, L, S) predicted future states
        """
        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x C
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc = x_enc / std_enc

        # ODE Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            x_rec, x_pred = self.ode_predictors[i](residual)
            residual = residual - x_rec  # Update residual
            if forecast is None:
                forecast = x_pred
            else:
                forecast += x_pred

        # De-stationarization
        forecast = forecast * std_enc + mean_enc

        return forecast
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise NotImplementedError(f"Task {self.task_name} not implemented")