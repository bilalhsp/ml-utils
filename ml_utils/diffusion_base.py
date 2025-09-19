import tqdm
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_utils import utils

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class DiffusionBaseEDM(ABC):
    def __init__(self, beta_min=0.0001, beta_max=0.02, T=1000, use_float64=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
        if use_float64:
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        self.betas = np.linspace(beta_min, beta_max, T)
        self.alphas = 1 - self.betas     # alpha = 1 - beta
        self.alpha_bars = np.cumprod(self.alphas)
        self.sigma_t = np.sqrt((1 - self.alpha_bars) / self.alpha_bars)

    def sigma_schedule(self, num_steps, rho=7, sigma_min=0.0064, sigma_max=80):
        if num_steps == 1:
            return np.array([sigma_max])
        step_indices = np.arange(num_steps)
        return (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho


    def condition_index(self, sigma):
        """Returns the condition index for the given sigma value.
        This is refered to as C_cond(sigma) in the EDM paper.
        DNN in iDDPM is conditioned on index of markov chain (t: 0-T) at which the sample is generated.
        For DDIM, they condition directory on the sigma value
        """
        pass

    def denoised_estimate(self, x_hat, sigma):
        """Returns denoised sample D_{\theta}(x, sigmal) as given in EDM paper.

        Args:
            x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
            sigma: noise level

        """
        pass

    def sigma_inv(self, sigma):
        """Returns the inverse of the function sigma(t)."""
        pass

class EDMiDDPM(nn.Module, DiffusionBaseEDM):
    """Implements the iDDPM model from 'Improved Denoising Diffusion Probabilistic Models' (Nichol et al. 2021).
    This is a denoising diffusion model that uses the noise prediction loss.
    """
    def __init__(self, estimator=None, beta_min=0.0001, beta_max=0.02, T=1000, use_float64=True):
        super().__init__()
        DiffusionBaseEDM.__init__(self, beta_min, beta_max, T)
        self.estimator = estimator
        if self.estimator is not None:
            self.estimator = self.estimator.to(self.device)

        

    def sample_xt(self, x0, sigma):
        x_sigma = x0 + sigma*torch.randn(x0.shape, dtype=self.dtype, device=x0.device)
        return x_sigma
    
    def estimate_epsilon(self, x, t):
        """Estimate the noise at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
        """
        
        return self.estimator(x.to(torch.float32), t.to(torch.float32))[:,:3].to(self.dtype)

    def iddpm_sigma_schedule(self, num_steps=1000, idx_offset=0):
        """Implements the iDDPM sigma schedule, by using linear steps from max_index-j0 to 0.
        Sigma schedule is defined by the value of sigma at each of these indices.

        EDM paper uses an offset j0 to start the schedule from a higher value of sigma.
        For cosine schedule, they used j0=8, which gives maximum value of sigma=80.
        For linear schedule, this is j0=69 , which gives maximum value of sigma=80.
        
        Args:
            num_steps: number of steps to generate the sample
            idx_offset: offset to start the schedule from, same as j0 in EDM paper.
        """
        step_indices = np.arange(num_steps)
        t_indices =  self.T - 1 - np.floor(idx_offset + (self.T-1-idx_offset)/(num_steps-1)*step_indices).astype(int) 
        return self.sigma_t[t_indices]
    
    def get_sigma_index(self, sigma, beta_min=0.1, beta_d=19.9):
        # sigma = torch.as_tensor(sigma).to(self.dtype)
        return ((beta_min ** 2 + 2 * beta_d * (1 + sigma ** 2).log()).sqrt() - beta_min) / beta_d


    def condition_index(self, sigma):
        """Returns the condition index for the given sigma value.
        This is refered to as C_cond(sigma) in the EDM paper.
        DNN in iDDPM is conditioned on index of markov chain (t: 0-T) at which the sample is generated.
        For DDIM, they condition directory on the sigma value
        """
        # return np.argmin(np.abs(self.sigma_t - sigma)) #  # -1 because we start from 0 index
        beta_d          = 19.9         # Extent of the noise level schedule.
        beta_min        = 0.1          # Initial slope of the noise level schedule.
        M               = 1000         # Original number of timesteps in the DDPM formulation.

        c_noise = (M - 1) * self.get_sigma_index(sigma, beta_min, beta_d)
        return c_noise
    
    def score_estimate(self, xt, sigma):
        """Estimate the score at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((xt.shape[0],), sigma, dtype=self.dtype, device=xt.device)
        c_noise = self.condition_index(sigma)
        c_in = 1 / torch.sqrt(sigma**2 + 1)

        c_in = utils.match_dimensions(c_in, xt.shape)
        eps = self.estimate_epsilon(c_in*xt, c_noise)
        denom = utils.match_dimensions(sigma, xt.shape)
        return - eps / denom


    def denoised_estimate(self, x_hat, sigma):
        """Returns denoised sample D_{\theta}(x, sigmal) as given in EDM paper.

        Args:
            x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
            sigma: noise level float or tensor of shape (N,)

        """
        if not torch.is_tensor(sigma):
            sigma = torch.full((x_hat.shape[0],), sigma, dtype=self.dtype, device=x_hat.device)
        c_noise = self.condition_index(sigma)
        # c_noise = torch.full((x_hat.shape[0],), c_noise, dtype=self.dtype, device=x_hat.device)
        c_in = 1 / torch.sqrt(sigma**2 + 1)
        c_out = -sigma

        c_in = utils.match_dimensions(c_in, x_hat.shape)
        c_out = utils.match_dimensions(c_out, x_hat.shape)
        eps = self.estimate_epsilon(c_in*x_hat, c_noise)
        return x_hat + c_out*eps
    
    def dx_estimate(self, x, sigma):
        """Returns the dx estimate D_{\theta}(x, sigmal) as given in EDM paper.
        This is used to compute the reverse diffusion step.

        Args:
            x_hat: (N, C, H, W) tensor, sample before scaling by s(t)
            sigma: noise level

        """
        # simplified expression for d_cur, for s(t) = 1, sigma(t) = t
        denoised_x = self.denoised_estimate(x, sigma)
        out = (x - denoised_x)/sigma
        return out
        
    
    def sigma_inv(self, sigma):
        """Returns the inverse of the function sigma(t)."""
        return sigma

    def get_sigma_t(self, t):
        """Returns the inverse of the function sigma(t)."""
        return t
    
    @torch.no_grad()
    def ddpm_ode_sample(
        self, n_evals=100, start_x=None, start_t=None, **kwargs
        ):

        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)

        shape = kwargs.get("shape", (1, 3, 256, 256))
        if start_t is not None:
            start_t = self.T
        offset = self.T - start_t
        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.iddpm_sigma_schedule(num_steps, offset)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        if start_x is not None:
            x = start_x.to(self.device, dtype=self.dtype)
        else:
            x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            d_cur = self.dx_estimate(x, sigma_current)

            # Euler step...
            x_prime = x + d_cur * (t_next - t_current)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next)
                x = x + 0.5 * (d_cur + d_prime) * (t_next - t_current)
        x = x.contiguous()
        return x
    
    @torch.no_grad()
    def edm_ode_sample(
        self, n_evals=100, start_x=None, **kwargs):
        rho = kwargs.get("rho", 7)
        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)

        sigma_min=kwargs.get("sigma_min", 0.0064)
        sigma_max=kwargs.get("sigma_max", 80)
        shape = kwargs.get("shape", (1, 3, 256, 256))

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.sigma_schedule(num_steps, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)
        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.
        if start_x is None:
            x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            x = start_x.to(self.device, dtype=self.dtype)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            d_cur = self.dx_estimate(x, sigma_current)

            # Euler step...
            x_prime = x + d_cur * (t_next - t_current)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next)
                x = x + 0.5 * (d_cur + d_prime) * (t_next - t_current)
        x = x.contiguous().to(torch.float32)
        return x
    

    @torch.no_grad()
    def ddpm_sde_sample(
        self, n_evals=100, shape=(1, 3, 256, 256), **kwargs):

        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)
        s_churn = kwargs.get("s_churn", 80)
        s_min = kwargs.get("s_min", 0.05)
        s_max = kwargs.get("s_max", 50)
        s_noise = kwargs.get("s_noise", 1.003)

        

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.iddpm_sigma_schedule(num_steps)

        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            epsilon = torch.randn_like(x) * s_noise

            if sigma_current > s_min and sigma_current < s_max:
                gamma = min(s_churn/num_steps, np.sqrt(2)-1)
                t_hat = t_current + gamma*t_current
                sigma_hat = self.get_sigma_t(t_hat)
                x_hat = x + epsilon*(sigma_hat**2 - sigma_current**2)**0.5
            else:
                gamma = 0.0
                t_hat = t_current
                sigma_hat = sigma_current
                x_hat = x 
        
            d_cur = self.dx_estimate(x_hat, sigma_hat)

            # Euler step...
            x_prime = x_hat + d_cur * (t_next - t_hat)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next)
                x = x_hat + 0.5 * (d_cur + d_prime) * (t_next - t_hat)
        x = x.contiguous()
        return x

    @torch.no_grad()
    def edm_sde_sample(
        self, n_evals=100, start_x=None, **kwargs):

        rho = kwargs.get("rho", 7)
        use_euler = kwargs.get("use_euler", False)
        show_progress = kwargs.get("show_progress", False)
        s_churn = kwargs.get("s_churn", 80)
        s_min = kwargs.get("s_min", 0.05)
        s_max = kwargs.get("s_max", 50)
        s_noise = kwargs.get("s_noise", 1.003)

        sigma_min=kwargs.get("sigma_min", 0.0064)
        sigma_max=kwargs.get("sigma_max", 80)
        shape = kwargs.get("shape", (1, 3, 256, 256))

        num_steps = n_evals if use_euler else n_evals//2
        sigma_values = self.sigma_schedule(num_steps, rho=rho, sigma_min=sigma_min, sigma_max=sigma_max)

        sigma_values = np.concatenate((sigma_values, np.zeros_like(sigma_values[:1])))  # adding last sigma value of 0.

        if start_x is None:
            x = sigma_values[0]*torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            x = start_x.to(self.device, dtype=self.dtype)

        for i, (sigma_current, sigma_next) in tqdm.tqdm(
                enumerate(zip(sigma_values[:-1], sigma_values[1:])),
                total=len(sigma_values)-1,
                desc="Generating...",
                disable=not show_progress
            ):
            
            # get values of time from sigma values
            t_current = self.sigma_inv(sigma_current)
            t_next = self.sigma_inv(sigma_next)

            epsilon = torch.randn_like(x) * s_noise

            if sigma_current > s_min and sigma_current < s_max:
                gamma = min(s_churn/num_steps, np.sqrt(2)-1)
                t_hat = t_current + gamma*t_current
                sigma_hat = self.get_sigma_t(t_hat)
                x_hat = x + epsilon*(sigma_hat**2 - sigma_current**2)**0.5
            else:
                gamma = 0.0
                t_hat = t_current
                sigma_hat = sigma_current
                x_hat = x 
        
            d_cur = self.dx_estimate(x_hat, sigma_hat)

            # Euler step...
            x_prime = x_hat + d_cur * (t_next - t_hat)

            if use_euler or sigma_next == 0:
                x = x_prime
            else:
                d_prime = self.dx_estimate(x_prime, sigma_next)
                x = x_hat + 0.5 * (d_cur + d_prime) * (t_next - t_hat)
        x = x.contiguous()
        return x
    
    @torch.no_grad()
    def generate(self, method, batch_size=1, **kwargs):
        """Generate samples using the diffusion model as a denoiser.
        Args:
            method: method to use for generation, e.g. 'ddpm', 'sde', etc.
            batch_size: number of samples to generate
            **kwargs: additional arguments for the generation method
        """
        shape = (batch_size, 3, 256, 256)
        if method == 'ddpm_ode':
            return self.ddpm_ode_sample(shape=shape, **kwargs)
        elif method == 'edm_ode':
            return self.edm_ode_sample(shape=shape, **kwargs)
        elif method == 'edm_sde':
            return self.edm_sde_sample(shape=shape, **kwargs)
        elif method == 'ddpm_sde':
            return self.ddpm_sde_sample(shape=shape, **kwargs)
        else:
            raise ValueError(f"Unknown generation method: {method}")



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
        # to avoid type cast to np.float64
        self.alpha_bars_prev = np.append(np.float32(1.0), self.alpha_bars[:-1])

        self.reverse_var = reverse_var

        if reverse_var =='beta_tilde':
            self.sigma_t = np.sqrt(self.betas * (1 - self.alpha_bars_prev) / (1 - self.alpha_bars))
        elif reverse_var == 'beta':
            self.sigma_t = np.sqrt(self.betas)
        else:
            raise ValueError(f"Unknown reverse variance parameter: {reverse_var}")
        
        


    ##############################################################################
    ###         Diffusion sampling as denoisers: starts here

    def match_dimensions(self, tensor, shape):
        """Return view of the input tensor to allow broadcasting with the shape."""
        tensor_shape = tensor.shape
        while len(tensor_shape) < len(shape):
            tensor_shape = tensor_shape + (1,) 
        return tensor.view(tensor_shape)
    

    def get_alpha_bar(self, t, s=None):
        """Get alpha_bar for transition from s to t. 
        t = [1, T], so appropriate adjust indices for zero-based index.
        
        Args:
            t: (N,) tensor of target time step
            s: (N,) tensor of initial, if None then treat as zero.
        """
        t = t - 1  # convert to zero-based index
        alpha_bar_t = torch.from_numpy(self.alpha_bars).to(t.device)[t.to(torch.int64)]
        if s is not None:
            s = s - 1  # convert to zero-based index
            alpha_bar_s = torch.from_numpy(self.alpha_bars).to(s.device)[s.to(torch.int64)]
            alpha_bar_t = alpha_bar_t / alpha_bar_s
        return alpha_bar_t

    def get_sigma_t(self, t, s=None):
        """
        Get sigma_t for transition from s to t. If s is None,
        then treat s = t-1 (i.e. sigma_t = sqrt(Beta_t)). 

        Args:
            t: (N,) tensor of target time step
            s: (N,) tensor of initial, if None then treat as zero.
        """
        if s is None:
            s = t - 1  # treat s as t-1 if not provided
        alpha_bar_st = self.get_alpha_bar(t, s)
        sigma_t = torch.sqrt(1 - alpha_bar_st)

        if self.reverse_var == 'beta_tilde':
            alpha_bar_s = self.get_alpha_bar(s)
            alpha_bar_t = self.get_alpha_bar(t)
            sigma_t = sigma_t * torch.sqrt((1 - alpha_bar_s) / (1 - alpha_bar_t))
        return sigma_t


    def score_estimate(self, x, t):
        """Estimate the score at time t, given x.
        Uses the relationship between the score and the noise prediction.
        
        Args:
            x: (N, C, H, W) tensor
            t: (N,) tensor of current time step
        """
        eps = self.estimator(x, t)[:,:3]
        denom = self.match_dimensions(torch.sqrt(1 - self.get_alpha_bar(t)), eps.shape)
        return - eps / denom #torch.sqrt(1 - self.get_alpha_bar(t)).view(-1, 1, 1, 1)
    
    def denoising_step(self, x, t, s=None, stoc=True):
        """Denoising step from x_t to x_s. 
        Predict p(x_s | x_t) using Tweedie's formula:

        Args:
            x: (N, C, H, W) tensor of current sample x_t at t.
            t: (N,) tensor of current time step.
            s: (N,) tensor of next time step, if None then treat as zero.
            stoc: whether to add stochastic term in the denoising step
        """

        score = self.score_estimate(x, t)
        alpha_st = self.get_alpha_bar(t, s)
        alpha_st = self.match_dimensions(alpha_st, x.shape)
        # tweedie's formula: for y = x + sqrt(1-alpha)*z, the denoised value is;
        x = x + (1 - alpha_st)*score
        # Normalize by alpha_bar to get the correct scale 
        # because noising process scales x by sqrt(alpha_bar)
        # y = torch.sqrt(alpha)*x + torch.sqrt(1 - alpha)*z
        x = x / torch.sqrt(alpha_st)

        if stoc and s is not None: # adds stochastic term
            z = torch.randn_like(x)
            sigma_st = self.get_sigma_t(t, s)
            sigma_st = self.match_dimensions(sigma_st, x.shape)
            x = x + sigma_st*z
        return x
    
    def ddpm_sample(
            self, n_timesteps=100, start_x=None, start_t=None, show_progress=True,
            **kwargs
            ):
        """Generates samples from the diffusion model treating it as a denoiser.
        We can also think of it as using shorter chain e.g. [1000, 900, 800, ...100, 0]
        to generate samples. The weights of tweedie's formula are adjusted accordingly:
        alpha_bar_st = alpha_bar_t / alpha_bar_s (gives weights for the transition between t and s)
        The variance of predicted sample at s is also adjusted accordingly.
        For 10 step generation, the last step is denoising to clean image (i.e. from t=100 to s=0).

        Args:
            n_timesteps: number of steps to generate the sample
            start_x: initial sample to start from, if None then random noise is used
            start_t: initial time step to start from, if None then T is used
            
        """
        if start_x is None:
            x = torch.randn(1, 3, 256, 256)
        else:
            x = start_x
        if start_t is None:
            start_t = self.T
        x = x.to(self.device)
        B = x.shape[0]
        h = start_t/n_timesteps     
        timesteps = [(start_t - (i + 0.0)*h) for i in range(n_timesteps)]
        # print(timesteps)
        for i, time_step in enumerate(tqdm.tqdm(timesteps, disable=not show_progress)):
            if i < (n_timesteps - 1):
                s = torch.full((B,), timesteps[i + 1], dtype=torch.float32, device=self.device)
            else:
                s = None    # final step, denoises it to clean image
            t = torch.full((B,), time_step, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                x = self.denoising_step(x, t, s=s, stoc=True)
        x = x.contiguous()
        return x
    
    @torch.no_grad()
    def ddim_sample(
            self, n_timesteps=100, start_x=None, start_t=None, show_progress=False, 
            use_dynamic_thresholding=True, **kwargs
            ):
        """Generates samples from the diffusion model using DDIM sampler.
        To sample p(x_s | x_t) it implements Eq. 7 from DDIM paper (Song 2021),
        with variance set to zero. 
        This corresponds to the deterministic sampling process with eta=0. 
        
        Args:
            n_timesteps: number of steps to generate the sample
            start_x: initial sample to start from, if None then random noise is used
            start_t: initial time step to start from, if None then T is used

        """
        if start_x is None:
            x = torch.randn(1, 3, 256, 256)
        else:
            x = start_x
        if start_t is None:
            start_t = self.T
        x = x.to(self.device)
        B = x.shape[0]
        h = start_t/n_timesteps     
        timesteps = [(start_t - (i + 0.0)*h) for i in range(n_timesteps)]
        for i, time_step in enumerate(tqdm.tqdm(timesteps, disable=not show_progress)): 
            if i < (n_timesteps - 1):
                s = torch.full((B,), timesteps[i + 1], dtype=torch.float32, device=self.device)
            else:
                s = None    # final step, denoises it to clean image
            t = torch.full((B,), time_step, dtype=torch.float32, device=self.device)
            x_0 = self.denoising_step(x, t, s=None, stoc=False)
            if s is None:
                x = x_0
            else:
                alpha_bar_s = self.get_alpha_bar(s)
                alpha_bar_s = self.match_dimensions(alpha_bar_s, x.shape)
                alpha_bar_t = self.get_alpha_bar(t)
                alpha_bar_t = self.match_dimensions(alpha_bar_t, x.shape)
                
                x = torch.sqrt(alpha_bar_s)*x_0 + torch.sqrt(1-alpha_bar_s)*(x - torch.sqrt(alpha_bar_t)*x_0)/(torch.sqrt(1-alpha_bar_t))
        x = x.contiguous()
        return x
    
    @torch.no_grad()
    def fast_ddim_sample(
            self, n_evals=100, start_x=None, start_t=None, max_sub_steps=2.5, **kwargs
            ):
        """Modified DDIM, specifically multiple steps are taken to compute x_0 
        at each time point. The idea is to improve the quality of x_s estimated
        at each time step, by taking multiple steps to estimate x_0.
        If this works out, we will be able to generate samples with fewer steps
        or even with same number of evaluations (NFEs) will be helpful for 
        conditional generation.
        
        Args:
            n_evals: number of function evaluations allowed for the generation.
            max_sub_steps: maximum number of steps to take per time step to estimate x_0.
                actual number of steps is decreased for smaller time steps. 
                ref. self.calculate_total_evaluations to know the exact schedule. 
            start_x: initial sample to start from, if None then random noise is used
            start_t: initial time step to start from, if None then T is used
        """
        if start_x is None:
            x = torch.randn(1, 3, 256, 256)
        else:
            x = start_x
        if start_t is None:
            start_t = self.T
        x = x.to(self.device)
        B = x.shape[0]

        # calculate n_timesteps for the given n_evals and max_sub_steps
        n_timesteps = self.calculate_n_timesteps(n_evals, max_sub_steps, start_t=start_t)
        h = start_t/n_timesteps     
        timesteps = [(start_t - (i + 0.0)*h) for i in range(n_timesteps)]
        Total_steps = 0
        # print(timesteps)
        for i, time_step in enumerate(tqdm.tqdm(timesteps)):
            if i < (n_timesteps - 1):
                s = torch.full((B,), timesteps[i + 1], dtype=torch.float32, device=self.device)
            else:
                s = None    # final step, denoises it to clean image
            t = torch.full((B,), time_step, dtype=torch.float32, device=self.device)
            
            # for n_timesteps=1, this gives prediction of x_0
            n_sub_steps = max(1, int(time_step*max_sub_steps/start_t))
            # n_sub_steps = max(1, int(max_sub_steps))
            Total_steps += n_sub_steps
            x_0 = self.ddim_sample(
                n_timesteps=n_sub_steps, start_x=x, start_t=time_step,
                show_progress=False
                )
            if s is None:
                x = x_0
            else:
                alpha_bar_s = self.get_alpha_bar(s)
                alpha_bar_s = self.match_dimensions(alpha_bar_s, x.shape)
                alpha_bar_t = self.get_alpha_bar(t)
                alpha_bar_t = self.match_dimensions(alpha_bar_t, x.shape)
                
                x = torch.sqrt(alpha_bar_s)*x_0 + torch.sqrt(1-alpha_bar_s)*(x - torch.sqrt(alpha_bar_t)*x_0)/(torch.sqrt(1-alpha_bar_t))
                # x = self.sample_xt(x_0, s)[0]
                # eps =  self.estimator(x, t)[:,:3]
                #  + torch.sqrt(1-alpha_bar_s)*eps
        x = x.contiguous()
        logging.info(f"Total steps taken: {Total_steps}")
        return x
    


    @torch.no_grad()
    def generate(self, method, batch_size=1, **kwargs):
        """Generate samples using the diffusion model as a denoiser.
        Args:
            method: method to use for generation, e.g. 'ddpm', 'sde', etc.
            batch_size: number of samples to generate
            **kwargs: additional arguments for the generation method
        """
        try:
            n_evals = kwargs.pop("n_evals")
        except KeyError:
            raise ValueError("The configuration must include a 'n_eval' key.")
        start_x = torch.randn((batch_size, 3, 256, 256), dtype=torch.float32)
        if method == 'ddpm':
            return self.ddpm_sample(n_timesteps=n_evals, start_x=start_x, **kwargs)
        elif method == 'ddim':
            return self.ddim_sample(n_timesteps=n_evals, start_x=start_x, **kwargs)
        elif method == 'fast_ddim':
            return self.fast_ddim_sample(n_evals=n_evals, start_x=start_x, **kwargs)
        else:
            raise ValueError(f"Unknown generation method: {method}")



    def get_signal_variance(self, t: torch.tensor):
        """"""
        return self.get_alpha_bar(t)
        # alpha_bar = torch.from_numpy(self.alpha_bars).to(t.device)[t.to(torch.int32)]
        # return alpha_bar



    @staticmethod
    def calculate_total_evaluations(n_timesteps, max_sub_steps, start_t=1000):
        """ Computes the total number of computations for the given repeats per step."""
        h = start_t/n_timesteps
        time_steps = np.array([start_t-(h * i) for i in range(n_timesteps)])
        repeats_per_step = [max(1, int(time_step*max_sub_steps/start_t)) for time_step in time_steps]
        return sum(repeats_per_step)
    
    @staticmethod
    def calculate_n_timesteps(n_evals, max_sub_steps, start_t=1000):
        """Calculates the number of timesteps required to achieve the given number of evaluations."""
        NFE = n_evals* max_sub_steps
        n_timesteps = n_evals
        while NFE > n_evals:
            n_timesteps = n_timesteps - 1
            NFE = HoDiffusion.calculate_total_evaluations(n_timesteps, max_sub_steps, start_t=start_t)
        logging.info(f"For NFEs={NFE}, n_timesteps={n_timesteps}")
        return n_timesteps

    

    ###         Diffusion sampling as denoisers: ends here
    ##############################################################################

    # Deprecated...
    # def get_signal_variance(self, t: torch.tensor):
    #     """"""
    #     alpha_bar = torch.from_numpy(self.alpha_bars).to(t.device)[t.to(torch.int32)]
    #     return alpha_bar
    
    def loss_t(self, x0, t):
        xt, z = self.sample_xt(x0, t)
        noise_estimation = self.estimator(xt, t)[:,:3]
        difference = noise_estimation - z
        loss = F.mse_loss(noise_estimation, z)
        return loss, xt, difference

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
        """Returns the noise level (Beta) at time t"""
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
    
    def reverse_ode(self, n_timesteps=100, start_x=None, start_t=None):
        """Sample from the diffusion model using stochastic diffusion sampling, when 
        modelled to predict noise epsilons at each time step.
        Args:
            use_xstart_pred: whether to use the predicted xstart to sample x_{t-1} or not.
        """
        if start_x is None:
            start_x = torch.randn(1, 3, 256, 256)
        x = start_x.to(self.device)
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
    

    def ode_euler_sampler(self, n_timesteps=100, start_x=None, start_t=None):
        """Sample from the diffusion model using stochastic diffusion sampling, when 
        modelled to predict noise epsilons at each time step.
        Args:
            use_xstart_pred: whether to use the predicted xstart to sample x_{t-1} or not.
        """
        if start_x is None:
            start_x = torch.randn(1, 3, 256, 256)
        x = start_x.to(self.device)
        if start_t is None:
            start_t = self.T
        start_t /= self.T
        h = start_t/n_timesteps
        sampling_times = self.T*np.array([(start_t - i*h) for i in range(n_timesteps)])
        for time_step in sampling_times:
            t = torch.full((x.shape[0],), time_step, dtype=torch.float32, device=self.device)

            # get the noise level (Beta) at time t
            noise_t = self.get_noise(t)
            noise_t = utils.match_tensor_dims(noise_t, x)

            with torch.no_grad():
                score_esimate = self.score_estimate(x, t)
                dxt = 0.5 * (x + score_esimate)*noise_t
                x = x + dxt*h
        
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
    
    
                                                 



    
    

        
