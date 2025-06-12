import tqdm
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionBase(ABC):
    def __init__(self, estimator, beta_min, beta_max, T):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.estimator = estimator
        if self.estimator is not None:
            self.estimator = self.estimator.to(self.device)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T


    @abstractmethod
    def get_signal_variance(self, t: torch.tensor):
        """For Ho.el al this is alpha_bar, for SDE this is torch.exp(-cum_noise)"""
        raise NotImplementedError
    
    def sample_xt(self, x0, t):
        """Forward diffusion step for SDE-based diffusion model.

        Args:
            x0: (N, C, H, W) or (N, C, L)
            t: (N,)
        """
        time = t
        while time.ndim < x0.ndim:
            time = time.unsqueeze(-1)
        alpha_bar = self.get_signal_variance(time) # alpha_bar
        mean = x0*torch.sqrt(alpha_bar)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(1-alpha_bar)
        return xt, z
    
    def get_SNR(self, t: torch.tensor):
        """Signal-to-Noise Ratio (SNR) at time t, defined as alpha_bar/(1-alpha_bar)"""
        alpha_bar = self.get_signal_variance(t)
        return alpha_bar/(1-alpha_bar)
    
class HoDiffusion(nn.Module, DiffusionBase):
    def __init__(
            self, estimator=None, beta_min=0.0001, beta_max=0.02, T=1000, schedule='linear', reverse_var='beta_tilde'
            ):
        super().__init__()
        DiffusionBase.__init__(self, estimator, beta_min, beta_max, T)
        if schedule=='linear':
            self.betas = np.linspace(self.beta_min, self.beta_max, self.T).astype(np.float32)
        
        elif schedule=='cosine':
            s = 0.008
            t = np.arange(T)
            f_t = (np.cos((t/T + s)*np.pi/(2+2*s)))**2
            alpha_bars = f_t/f_t[0]
            alpha_bars_prev = np.append(1, alpha_bars[:-1])
            self.betas = np.clip(1 - alpha_bars/alpha_bars_prev, 0, 0.999)

        self.alphas = 1 - self.betas     # alpha = 1 - beta
        self.alpha_bars = np.cumprod(self.alphas)
        self.alpha_bars_prev = np.append(1, self.alpha_bars[:-1])

        if reverse_var =='beta_tilde':
            self.sigma_t = np.sqrt(self.betas * (1 - self.alpha_bars_prev) / (1 - self.alpha_bars))
        elif reverse_var == 'beta':
            self.sigma_t = np.sqrt(self.betas)
        else:
            raise ValueError(f"Unknown reverse variance parameter: {reverse_var}")
        



    ###############################

    def get_signal_variance(self, t: torch.tensor):
        """"""
        alpha_bar = torch.from_numpy(self.alpha_bars).to(t.device)[t.to(torch.int32)]
        return alpha_bar
    
    def loss_t(self, x0, t):
        xt, z = self.sample_xt(x0, t)
        noise_estimation = self.estimator(xt, t)[:,:3]
        loss = F.mse_loss(noise_estimation, z)
        return loss, xt

    def compute_loss(self, x0):
        t = torch.randint(0, self.T, (x0.shape[0],), dtype=torch.int64,
                          device=x0.device, requires_grad=False)
        return self.loss_t(x0, t)
    
    ### Need to re-implement the following functions
    def reverse_diffusion(self, x=None, start_t=None, use_xstart_pred=True):
        """Sample from the diffusion model using stochastic diffusion sampling, when 
        modelled to predict noise epsilons at each time step.
        Args:
            use_xstart_pred: whether to use the predicted xstart to sample x_{t-1} or not.
        """
        if x is None:
            x = torch.randn(1, 3, 256, 256)
        x = x.to(self.device)
        if start_t is None:
            start_t = self.T

        for time_step in tqdm.tqdm(range(start_t)[::-1]):
            t = torch.tensor([time_step]).to(self.device)
            with torch.no_grad():
                eps = self.estimator(x, t)[:,:3]        # for guided diffusion model..
                if use_xstart_pred:
                    x = self.sample_x_prev_using_xstart_pred(x, t, eps)
                else:
                    x = self.sample_x_prev(x, t, eps)
        x = x.contiguous()
        return x
    
    def posterior_mean_using_xstart_pred(self, x, t, x_start):
        """Predict the posterior mean from the (predicted) xstart
        implementes Eq. 7 in Ho et al. (2020)
        """
        coef1 = self.betas[t] * np.sqrt(self.alpha_bars_prev[t]) / (1.0 - self.alpha_bars[t])
        coef2 = (1.0 - self.alpha_bars_prev[t]) * np.sqrt(self.alphas[t]) / (1.0 - self.alpha_bars[t])
        
        return (coef1 * x_start + coef2 * x)
    
    def predict_xstart_from_eps(self, x, t, eps):
        """predict x0 from x_t and eps_t, using the parameterization of q(x_t/x_0) as;
            xt(x0; eps) = np.sqrt(alpha_bar)*x0 + np.sqrt(1-alpha_bar)*eps
        given in paragraph underneath Eq. 8 in Ho et al. (2020) paper. Re-arranging this
        gives prediction of x_start as;
        x0 = np.sqrt(1/alpha_bar)*x - np.sqrt((1/alpha_bar) - 1)*eps
        """
        return np.sqrt(1.0 / self.alpha_bars[t]) * x - np.sqrt((1.0 / self.alpha_bars[t]) - 1) * eps


    def sample_x_prev_using_xstart_pred(self, x, t, eps):
        """Sample from p(x_{t-1} | x_t) by first predicting x0. Here are the steps;
        
        - Given (x_t, eps, t) predict x0 (using reparametrization of q(x_t/x_0))
        - clip x0 to be within [-1, 1]
        - Predict the posterior mean from x0 (using Eq. 7 in Ho et al. (2020))
        """
        pred_xstart = self.predict_xstart_from_eps(x, t, eps)
        # clip x_start to be within [-1, 1]
        pred_xstart = pred_xstart.clamp(-1, 1)
        post_mean = self.posterior_mean_using_xstart_pred(x, t, pred_xstart)

        # sample from p(x_{t-1} | x_t)
        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        sample = post_mean + self.sigma_t[t] * z
        return sample
    
    def sample_x_prev(self, x, t, eps):
        """Sample from p(x_{t-1} | x_t) by calculating the posterior mean using;
        post_mean = 1/sqrt(alpha) * (x - beta/sqrt(1 - alpha_bar) * eps)
        as given by Eq. 11 in Ho et al. (2020).        
        """
        coef1 = self.betas[t] / np.sqrt(1 - self.alpha_bars[t])
        coef2 = 1/np.sqrt(self.alphas[t]) 
        post_mean = coef2*(x - coef1 * eps)

        if t > 0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        sample = post_mean + self.sigma_t[t] * z
        return sample
    
    def get_noise(self, t):
        # noise = self.beta_min + (self.beta_max - self.beta_min)*t
        
        t = t/self.T
        noise = 0.05 + (20 - 0.05)*t
        return noise
    
    @torch.no_grad()
    def reverse_diffusion_step(self, z, t, h, stoc=False):
        """One step of reverse diffusion at time t."""
        xt = z
        time = t
        while time.ndim < z.ndim:
            time = time.unsqueeze(-1)
        noise_t = self.get_noise(time)
        if stoc:  # adds stochastic term
            dxt_det = -0.5 * xt - self.estimator(xt,t)
            dxt_det = dxt_det * noise_t * h
            dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                    requires_grad=False)
            dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
            dxt = dxt_det + dxt_stoc
        else:
            alpha_bar = self.get_signal_variance(time.to(torch.int32))
            eps = self.estimator(xt, t)[:,:3] 
            score = -eps/torch.sqrt(1 - alpha_bar)
            dxt = 0.5 * ( - xt - score)
            dxt = dxt * noise_t * h
        xt = (xt - dxt)
        return xt
    
    def reverse_ode(self, x=None, start_t=None, n_timesteps=100):
        """Sample from the diffusion model using stochastic diffusion sampling, when 
        modelled to predict noise epsilons at each time step.
        Args:
            use_xstart_pred: whether to use the predicted xstart to sample x_{t-1} or not.
        """
        if x is None:
            x = torch.randn(1, 3, 256, 256)
        x = x.to(self.device)
        if start_t is None:
            start_t = self.T
        start_t /= self.T
        h = start_t/n_timesteps
        sampling_times = self.T*np.array([(start_t - (i + 0.5)*h) for i in range(n_timesteps)])

        for t in sampling_times:
            t = t* torch.ones(
                x.shape[0], dtype=torch.int32, device=self.device
                )
            x = self.reverse_diffusion_step(x, t, h)
        x = x.contiguous()
        return x



class SDEDiffusion(nn.Module, DiffusionBase):
    """Stochastic Differential Equation (SDE) based diffusion model, as proposed in 
    Song et al. (2021) 'SCORE-BASED GENERATIVE MODELING THROUGH SDEs' (All stars 2021)
    and used by 
    Popov et al. (2021) 'Grad-TTS: A Diffusion Probabilistic Model for Text-to-Speech' 
    t \in [0, T].
    """
    def __init__(self, estimator=None, beta_min=0.05, beta_max=20, T=1):
        super().__init__()
        DiffusionBase.__init__(self, estimator, beta_min, beta_max, T)


    def get_signal_variance(self, t):
        noise = self.beta_min*t + 0.5*(self.beta_max - self.beta_min)*(t**2)
        return torch.exp(-noise)
    
    def get_noise(self, t):
        noise = self.beta_min + (self.beta_max - self.beta_min)*t
        return noise
    
    def loss_t(self, x0, t):
        xt, z = self.sample_xt(x0, t)
        time = t
        while time.ndim < x0.ndim:
            time = time.unsqueeze(-1)

        alpha_bar = self.get_signal_variance(time) # alpha_bar
        noise_estimation = self.estimator(xt, t)
        noise_estimation *= torch.sqrt(1.0 - alpha_bar)
        loss = torch.sum((noise_estimation + z)**2) / (x0.numel())
        return loss, xt

    def compute_loss(self, x0, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, t)
    
    @torch.no_grad()
    def reverse_diffusion(self, z, n_timesteps=100, start_t=1.0, stoc=False):
        h = start_t / n_timesteps
        xt = z #* mask
        for i in range(n_timesteps):
            t = (start_t - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype, 
                                                 device=z.device)
            time = t
            while time.ndim < z.ndim:
                time = time.unsqueeze(-1)
            noise_t = self.get_noise(time)
            if stoc:  # adds stochastic term
                dxt_det = -0.5 * xt - self.estimator(xt,t)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                # print(xt.shape, t.shape)
                dxt = 0.5 * ( - xt - self.estimator(xt, t.view(z.shape[0])))
                dxt = dxt * noise_t * h
            xt = (xt - dxt)
        return xt
    
    @torch.no_grad()
    def reverse_diffusion_step(self, z, t, h, stoc=False):
        """One step of reverse diffusion at time t."""
        xt = z
        time = t
        while time.ndim < z.ndim:
            time = time.unsqueeze(-1)
        noise_t = self.get_noise(time)
        if stoc:  # adds stochastic term
            dxt_det = -0.5 * xt - self.estimator(xt,t)
            dxt_det = dxt_det * noise_t * h
            dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                    requires_grad=False)
            dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
            dxt = dxt_det + dxt_stoc
        else:
            dxt = 0.5 * ( - xt - self.estimator(xt, t))
            dxt = dxt * noise_t * h
        xt = (xt - dxt)
        return xt
    
    
                                                 



    
    

        
