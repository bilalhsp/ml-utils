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

class Wav2Vec2Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_data,
        eval_data=None,
        diffusion_config=None,
        **kwargs,
        ):

        super(Wav2Vec2Trainer, self).__init__(
            model, 
            train_data,
            eval_data=eval_data,
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
        if self.distributed:
            self.noise_steps = self.model.module.get_noise_steps()
        else:
            self.noise_steps = self.model.get_noise_steps()

    def forward_step(self, batch):

        input_values = batch['input_values'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        B, T = input_values.shape
        t = np.random.randint(0, self.noise_steps, (B,))
        x_t, _ = self.diffusion.sample_xt(input_values, t)

        loss, logits = self.model(x_t, torch.from_numpy(t).to(self.device), attention_mask=attention_mask, labels=labels)
        return loss
    
    def evaluation_step(self, batch):
        input_values = batch['input_values'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)

        B, T = input_values.shape
        t = np.random.randint(0, self.noise_steps, (B,))
        x_t, _ = self.diffusion.sample_xt(input_values, t)
        if self.distributed:
            out = self.model.module.evaluate(x_t, torch.from_numpy(t).to(self.device), attention_mask=attention_mask, labels=labels)
        else:
            out = self.model.evaluate(x_t, torch.from_numpy(t).to(self.device), attention_mask=attention_mask, labels=labels)
        return {'eval_loss': out['wer']}



