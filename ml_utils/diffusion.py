import math
import tqdm
import torch
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt

class Diffusion:
    def __init__(self, diff_net=None, T=1000, beta_0=0.0001, beta_T=0.02, schedule='linear',
                  reverse_var='beta_tilde'):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if diff_net is not None:
            self.diff_net = diff_net.to(self.device)
        else:
            self.diff_net = diff_net
        # self.classifier = classifier.to(self.device)

        self.num_steps = T
        self.beta_start = beta_0
        self.beta_end = beta_T

        if schedule=='linear':
            self.betas = np.linspace(self.beta_start, self.beta_end, self.num_steps).astype(np.float32)
        
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


    
    def denoise_step(self, x_t, t, requires_grad=False):
        if requires_grad:
            out = self.diff_net(x_t, t)
        else:
            with torch.no_grad():
                out = self.diff_net(x_t, t)
        return out[:,:3] 
    
    def sample_alpha_bar(self, B=1):
        t = np.random.randint(0, self.num_steps, B, dtype=int)
        ls = np.sqrt(self.alpha_bars[t])
        ls_1 = np.sqrt(self.alpha_bars_prev[t])
        alpha_bar = np.random.uniform(ls_1, ls, B)**2
        return alpha_bar.astype(np.float32)
        

    def sample_xt_using_noise_level(self, x_0, alpha_bar):
        """Sample x_t using x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*eps_t

        Args:
            x_0: (N, C, H, W)
            t: (N,)
        """
        B = x_0.shape[0]
        alpha_shape = (B, *(1 for _ in range(x_0.ndim - 1)))
        # alpha_bar = torch.from_numpy(self.alpha_bars[t]).view(alpha_shape).to(self.device)
        alpha_bar = torch.from_numpy(alpha_bar).view(alpha_shape).to(self.device)
        eps = torch.randn_like(x_0)

        x_t = torch.sqrt(alpha_bar)*x_0 + torch.sqrt(1 - alpha_bar)*eps
        # x_t = np.sqrt(self.alpha_bars[t])*x_0 + np.sqrt(1-self.alpha_bars[t])*eps
        return x_t, eps

    def sample_xt(self, x_0, t):
        """Sample x_t using x_t = sqrt(alpha_bar)*x_0 + sqrt(1-alpha_bar)*eps_t

        Args:
            x_0: (N, C, H, W)
            t: (N,)
        """
        alpha_shape = (x_0.shape[0], *(1 for _ in range(x_0.ndim - 1)))
        alpha_bar = torch.from_numpy(self.alpha_bars[t]).view(alpha_shape).to(self.device)

        eps = torch.randn_like(x_0)

        x_t = torch.sqrt(alpha_bar)*x_0 + torch.sqrt(1 - alpha_bar)*eps
        # x_t = np.sqrt(self.alpha_bars[t])*x_0 + np.sqrt(1-self.alpha_bars[t])*eps
        return x_t, eps


    def predict_xstart_from_eps(self, x, t, eps):
        """predict x0 from x_t and eps_t, using the parameterization of q(x_t/x_0) as;
            xt(x0; eps) = np.sqrt(alpha_bar)*x0 + np.sqrt(1-alpha_bar)*eps
        given in paragraph underneath Eq. 8 in Ho et al. (2020) paper. Re-arranging this
        gives prediction of x_start as;
        x0 = np.sqrt(1/alpha_bar)*x - np.sqrt((1/alpha_bar) - 1)*eps
        """
        return np.sqrt(1.0 / self.alpha_bars[t]) * x - np.sqrt((1.0 / self.alpha_bars[t]) - 1) * eps

    def posterior_mean_using_xstart_pred(self, x, t, x_start):
        """Predict the posterior mean from the (predicted) xstart
        implementes Eq. 7 in Ho et al. (2020)
        """
        coef1 = self.betas[t] * np.sqrt(self.alpha_bars_prev[t]) / (1.0 - self.alpha_bars[t])
        coef2 = (1.0 - self.alpha_bars_prev[t]) * np.sqrt(self.alphas[t]) / (1.0 - self.alpha_bars[t])
        
        return (coef1 * x_start + coef2 * x)


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
    
     


    def sample_using_SDE(self, x=None, start_t=None, use_xstart_pred=True):
        """Sample from the diffusion model using stochastic diffusion sampling, when 
        modelled to predict noise epsilons at each time step.
        Args:
            model_fn: the diffusion model that predicts epsilons for a given input and time step.
            use_xstart_pred: whether to use the predicted xstart to sample x_{t-1} or not.
        """
        if x is None:
            x = torch.randn(1, 3, 256, 256).to(self.device)
        else:
            x = x.to(self.device)
        if start_t is None:
            start_t = self.num_steps
        for time_step in tqdm.tqdm(range(start_t)[::-1]):
            t = torch.tensor([time_step]).to(self.device)
            # eps = model_fn(x, t)
            eps = self.denoise_step(x, t)
            if use_xstart_pred:
                x = self.sample_x_prev_using_xstart_pred(x, t, eps)
            else:
                x = self.sample_x_prev(x, t, eps)

        x = x.contiguous()
        return x

    def inverse(self, diff_net=None, x=None, start_t=None, use_xstart_pred=True, model_fn=None):
        """Sample from the diffusion model using stochastic diffusion sampling, when 
        modelled to predict noise epsilons at each time step.
        Args:
            model_fn: the diffusion model that predicts epsilons for a given input and time step.
            use_xstart_pred: whether to use the predicted xstart to sample x_{t-1} or not.
        """
        if x is None:
            x = torch.randn(1, 3, 256, 256)
        x = x.to(self.device)
        if start_t is None:
            start_t = self.num_steps

        for time_step in tqdm.tqdm(range(start_t)[::-1]):
            t = torch.tensor([time_step]).to(self.device)
            if model_fn is not None:
                if diff_net is None:
                    model = self.diff_net
                else:
                    model = diff_net 
                eps = model_fn(x, t)
            else:
                if diff_net is None:
                    eps = self.denoise_step(x, t)
                else:
                    eps = diff_net(x,t)
            if use_xstart_pred:
                x = self.sample_x_prev_using_xstart_pred(x, t, eps)
            else:
                x = self.sample_x_prev(x, t, eps)

        # x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        # x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        return x

    def predict_x_prev_for_DDIM(self, x, t, eps):
        """Predicts previous sample x_{t-1} from current sample x_t and noise eps_t
        using DDIM method.
        """
        term1 = x - np.sqrt(1 - self.alpha_bars[t])*eps
        term1_multiplier = np.sqrt(self.alpha_bars_prev[t])/ np.sqrt(self.alpha_bars[t])
        term2 = np.sqrt(1-self.alpha_bars_prev[t]) * eps
        return term1_multiplier*term1 + term2
    
    def guided_diffusion(self, model_fn, cond_fn=None, x=None, y=None, s=1, use_noisy_cond=True):
        """Guided diffusion sampling, where the model_fn is the diffusion model
        and cond_fn is the conditional function that takes in an input and a target

        Args:
            model_fn: the diffusion model that predicts epsilons for a given input and time step.
            cond_fn: the conditional function that takes in an input and target and returns the gradient.
            y: the target class for the conditional function.
            s: the scaling factor for the conditional gradient.
            use_noisy_cond: whether to use the noisy conditional gradient or not.
        """
        if x is None:
            x = torch.randn(1, 3, 256, 256).to(self.device)
        else:
            x = x.to(self.device)
        for time_step in tqdm.tqdm(range(self.num_steps)[::-1]):
            t = torch.tensor([time_step]).to(self.device)
            eps = model_fn(x, t)
            if cond_fn is not None:
                if use_noisy_cond:
                    cond_grad = s*cond_fn(x, y, t)
                else:
                    cond_grad = s*cond_fn(x, y)

                grad_multiplier = np.sqrt(1-self.alpha_bars[t])
                eps = eps - grad_multiplier * cond_grad

            x = self.predict_x_prev_for_DDIM(x, t, eps)


        # x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        # x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
        return x

    def sample_using_DDIM(self, x=None, cond_fn=None, y=None, s=1, use_noisy_cond=True,
                          grad_multiplier=None, display_every=100, start_t=None, magic_t=1000,
                          dead_multiplier=1, repetitions=1, loop_idx=200, init_steps=10,
                          early_stopping=False, cosine_diff=False, **kwargs
                          ):
        """Guided diffusion sampling, where the model_fn is the diffusion model
        and cond_fn is the conditional function that takes in an input and a target

        Args:
            model_fn: the diffusion model that predicts epsilons for a given input and time step.
            cond_fn: the conditional function that takes in an input and target and returns the gradient.
            y: the target class for the conditional function.
            s: the scaling factor for the conditional gradient.
            use_noisy_cond: whether to use the noisy conditional gradient or not.
        """
        if x is None:
            x = torch.randn(1, 3, 256, 256).to(self.device)
        else:
            x = x.to(self.device)
        if start_t is None:
            start_t = self.num_steps
        # x = x*(start_t/self.num_steps)
        for _ in range(init_steps):
            x = x + cond_fn(x, y)[0]
        for idx, time_step in enumerate(tqdm.tqdm(range(start_t)[::-1])):
            if cosine_diff and idx < 250:
                t = ((time_step)/1.5 + (time_step)/3*math.cos((idx)/40))#/steps*total + offset # Linearly decreasing + cosine
                t = torch.tensor([int(t)]).to(self.device)
            else:
                t = torch.tensor([time_step]).to(self.device)
            if idx < loop_idx:
                loop_over = repetitions
            else:
                loop_over = 1
            for _ in range(loop_over):
                eps = self.denoise_step(x, t)
                if cond_fn is not None:
                    
                    if use_noisy_cond:
                        cond_grad, _ = cond_fn(x, y, t)
                    else:
                        pred_xstart = self.predict_xstart_from_eps(x, t, eps)
                        cond_grad, _ = cond_fn(pred_xstart, y)

                    if grad_multiplier is None:
                        grad_multiplier = np.sqrt(1-self.alpha_bars[t])
                    if time_step > magic_t:
                        s = dead_multiplier
                    # original multiplier   
                    eps = eps - grad_multiplier * s*cond_grad

                x = self.predict_x_prev_for_DDIM(x, t, eps)
            if early_stopping is not None:
                if t.item() < early_stopping:
                    break
            if (t.item()+1) %display_every==0 or t.item()==0:
                with torch.no_grad():
                    fig, ax = plt.subplots(1,1, figsize=(10,5))
                    ax.imshow(0.5*(x+1)[0].cpu().numpy().transpose(1,2,0))
                    ax.set_title("Infered Image")
                    plt.show()
        x = x.contiguous()
        return x


    def sample_using_GD(self, model_fn, cond_fn=None, y=None, lr=0.1, lmbda=0.1, T=10):
        
        x = torch.randn(1, 3, 256, 256).to(self.device)
        for time_step in tqdm.tqdm(range(self.num_steps)[::-T]):
            t = torch.tensor([time_step]).to(self.device)
            eps = model_fn(x, t)
            cond_grad = cond_fn(x, y)

            diff_score = -eps/np.sqrt(1-self.alpha_bars[t])
            combined_score = lr*cond_grad + lmbda*diff_score 

            # alpha_i = lr*self.betas[t]/self.betas[0] 
            # for _ in range(T):
            #     eps = model_fn(x, t)
            #     cond_grad = cond_fn(x, y)

            #     diff_score = -eps/np.sqrt(1-self.alpha_bars[t])
                # combined_score = cond_grad + lmbda*diff_score 

                # combined_score = diff_score 

                # Annealed Langevin dynamics alg. 1 from Song & Ermon (2019)
                    
                # x = x + alpha_i*combined_score/2 + np.sqrt(alpha_i)*torch.randn_like(x)

            mean = combined_score.mean(dim=(2, 3), keepdim=True)  # Compute mean over H and W
            std = combined_score.std(dim=(2, 3), keepdim=True)    # Compute std over H and W
            combined_score = (combined_score - mean) / (std + 1.e-8)     # Normalize each channel

            
            # combined_score = combined_score.clamp(-1, 1)
            x = x + combined_score

            # x = (x - x.mean()) / x.std()

            mean = x.mean(dim=(2, 3), keepdim=True)  # Compute mean over H and W
            std = x.std(dim=(2, 3), keepdim=True)    # Compute std over H and W
            x = (x - mean) / (std + 1.e-8)     # Normalize each channel


            # x = x.clamp(-1, 1)

        return x

    # def sample_using_variational_inference(self, diffusion_grad, cond_fn, y, lmbda):
    #     """Sample from the diffusion model using variational inference,
    #     """
    #     x = torch.randn(1, 3, 256, 256).to(self.device)
    #     y = torch.tensor(y).to(x.device)
    #     for time_step in tqdm.tqdm(range(self.num_steps)[::-1]):
    #         t = torch.tensor([time_step]).to(self.device)
    #         eps = torch.randn(1, 3, 256, 256).to(self.device)

    #         x_in = x.detach().requires_grad_(True)
    #         time_step = torch.tensor([1]).to(x.device)
    #         logits = classifier.to(x.device)(x_in, time_step)
    #         log_probs = F.log_softmax(logits, dim=-1)
    #         selected = log_probs[range(len(logits)), y.view(-1)]

    #         x_t = np.sqrt(self.alpha_bars[t])*x_in + np.sqrt(1-self.alpha_bars[t])*eps

    #         out = model.to(x.device)(x_t, t)
    #         pred_eps, _ = torch.split(out, 3, dim=1)
    #         error_norm = torch.sum((eps - pred_eps)**2) - selected.sum()
    #         grad = torch.autograd.grad(error_norm, x_in)[0]

    #         x = x - lmbda*grad

    #     return x
    

    def sample_using_variational_inference(
            self, model, cond_fn, y=417, lr=1, steps=200, attr_batch_size=8, 
            lr_start_factor=1, lr_end_factor=1, stop_after=None, s = 0.5,
            cosine_schedule=False, noisy_schedule=False, grad_multiplier=None,
            scaling_function=None, offset=200, total=800, **kwargs):

        if stop_after is None:
            stop_after = steps

        # if model is None:
        #     model = InferenceModel()
        model = model.to(self.device)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     opt,
        #     milestones=kwargs['milestones'],
        #     gamma=kwargs['gamma']
        #     )
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=lr_start_factor, end_factor=lr_end_factor, total_iters=steps
            )
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(
            opt,
            gamma=kwargs['gamma']
            )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            opt,
            schedulers=[scheduler1, scheduler2],
            milestones=kwargs['milestones'],
            )

        self.diff_net.eval()
        # attr_model.eval()

        norm_track = 0
        bar = tqdm.tqdm(range(steps))
        losses = []
        update_every = 10*steps/200
        display_every = 25*steps/200
        loss_history = {}
        for i, _ in enumerate(bar):
            if i > stop_after:
                print('Early stopping...')
                break
            # Select t   
            # total = 1000
            
            # total = 1000 - offset
            if cosine_schedule:       
                t = ((steps-i)/1.5 + (steps-i)/3*math.cos(i/10))/steps*total + offset # Linearly decreasing + cosine
            else:
                t = ((steps-i))/steps*total + offset # Linearly decreasing
            if noisy_schedule:
                t = np.array([t + np.random.randint(-50, 51) for _ in range(1)]).astype(int) # Add noise to t
            else:
                t = np.array([t]).astype(int) # Add noise to t
            t = np.clip(t, 0, 999)

            if grad_multiplier is None:
                    grad_multiplier = np.sqrt(1-self.alpha_bars[t[0]])

            if scaling_function is not None:
                scale = scaling_function(steps, t[0])

            # Denoising
            sample_img = model.encode()
            xt, epsilon = self.sample_xt(sample_img, t)
            t = torch.from_numpy(t).float().view(1)

            epsilon_pred = self.denoise_step(xt.float(), t.to(self.device), requires_grad=True)
            # compute diffuson loss
            loss = F.mse_loss(epsilon, epsilon_pred)
            opt.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad_norm = torch.linalg.norm(model.img.grad)

                if i > 0:
                    alpha = 0.5
                    norm_track = alpha*norm_track + (1-alpha)*grad_norm
                else:
                    norm_track = grad_norm

            opt.step()

            opt.zero_grad()
            # loss.backward()
            grad, loss = cond_fn(model.encode(), y)
            
            # model.img.grad = -grad_multiplier*s*grad
            model.img.grad = -grad
            loss = -1*loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), s*norm_track)
            opt.step()
            scheduler.step()

            losses.append(loss.item())

            if i% 25 ==0:
                bar.set_postfix({'loss': np.mean(losses)})
                loss_history[t.item()] = np.mean(losses)
                losses = []

            if (i+1) %25==0 or i==0:
                with torch.no_grad():
                    fig, ax = plt.subplots(1,1, figsize=(10,5))
                    ax.imshow(0.5*(model.encode()+1)[0].cpu().numpy().transpose(1,2,0))
                    ax.set_title("Infered Image")
                    plt.show()

        
        return model, loss_history


