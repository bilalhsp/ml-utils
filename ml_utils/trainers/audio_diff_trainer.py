import logging
import torch
import numpy as np
import torchvision
import torch.distributed as dist

from .base_trainer import BaseTrainer
from ml_utils import Diffusion

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

class AudioDiffTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        train_data,
        diffusion_config=None,
        **kwargs,
        ):

        super(AudioDiffTrainer, self).__init__(
            model, 
            train_data,
            criterion=criterion, 
            **kwargs,
        )
        # Default diffusion configuration
        if diffusion_config is None:
            diffusion_config = {
                'T': 1000,  # or some default values you want
                'beta_0': 0.0001,
                'beta_T': 0.02
            }

        # Instantiate Diffusion with the provided or default config
        self.diffusion = Diffusion(**diffusion_config)

    def forward_step(self, batch):
        audios, labels = batch
        audios = audios.to(self.device)
        B, C, L = audios.shape

        alpha_bar = self.diffusion.sample_alpha_bar(B)
        x_t, eps = self.diffusion.sample_xt_using_noise_level(audios, alpha_bar)
        model_input = (x_t, torch.from_numpy(alpha_bar).to(self.device).view(B,1))

        # t = np.random.randint(0, self.diffusion.num_steps, B, dtype=int)
        # x_t, eps = self.diffusion.sample_xt(audios, t)
        # model_input = (x_t, torch.from_numpy(t).to(self.device).view(B,1))
        epsilon_pred = self.model(model_input)
        loss = self.criterion(epsilon_pred, eps)
        return loss



    # def training_step(self, batch, step):
    #     audios, labels = batch
    #     audios = audios.to(self.device)
    #     B, C, L = audios.shape

    #     alpha_bar = self.diffusion.sample_alpha_bar(B)
    #     x_t, eps = self.diffusion.sample_xt_using_noise_level(audios, alpha_bar)
    #     model_input = (x_t, torch.from_numpy(alpha_bar).to(self.device).view(B,1))

    #     # t = np.random.randint(0, self.diffusion.num_steps, B, dtype=int)
    #     # x_t, eps = self.diffusion.sample_xt(audios, t)
    #     # model_input = (x_t, torch.from_numpy(t).to(self.device).view(B,1))
    #     epsilon_pred = self.model(model_input)
    #     loss = self.criterion(epsilon_pred, eps)

    #     # self.optimizer.zero_grad()
    #     loss.backward()
    #     # self.optimizer.step()
    #     if (step+1) % self.gradient_accumulation_steps == 0 or (step+1) == len(self.train_dataloader):
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()

    #     return {'train_loss': loss.item()}    

    # def evaluation_step(self, batch):
    #     audios, labels = batch
    #     audios = audios.to(self.device)
    #     B, C, L = audios.shape

    #     # t = np.random.randint(0, self.diffusion.num_steps, B, dtype=int)
    #     # x_t, eps = self.diffusion.sample_xt(audios, t)
    #     # model_input = (x_t, torch.from_numpy(t).to(self.device).view(B,1))

    #     alpha_bar = self.diffusion.sample_alpha_bar(B)
    #     x_t, eps = self.diffusion.sample_xt_using_noise_level(audios, alpha_bar)
    #     model_input = (x_t, torch.from_numpy(alpha_bar).to(self.device).view(B,1))

    #     epsilon_pred = self.model(model_input)
    #     loss = self.criterion(epsilon_pred, eps)
    #     return {'eval_loss': loss.item()}
    
    # def evaluate(self, current_step):
    #     self.model.eval()
    #     if self.eval_dataloader is not None:
    #         eval_losses = []
    #         for batch in self.eval_dataloader:
    #             out = self.evaluation_step(batch)
    #             eval_losses.append(out['eval_loss'])
            
    #         if self.distributed:
    #             # aggregate losses across all devices
    #             losses_tensor = torch.tensor(eval_losses, device=self.device)
    #             dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
    #             mean_loss = torch.mean(losses_tensor).item()/dist.get_world_size()

    #         else:
    #             mean_loss = np.mean(eval_losses)

    #         if self.is_main_process():
    #             self.writer.add_scalar(
    #                 "Loss/eval",
    #                 mean_loss,
    #                 current_step
    #                 )

