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

class MNISTTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        train_data,
        eval_data=None,
        lr=1.e-3,
        output_dir='./trainer_output',
        exp_name=None,
        distributed=False,
        ):

        super(MNISTTrainer, self).__init__(
            model, 
            criterion, 
            train_data,
            eval_data,
            lr,
            output_dir,
            exp_name,
            distributed,
        )


    def training_step(self, batch):
        train_x, train_y = batch

        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)

        pred_y = self.model(train_x)
        loss = self.criterion(pred_y, train_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'train_loss': loss.item()}    

    def evaluation_step(self, batch):
        val_x, val_y = batch

        val_x = val_x.to(self.device)
        val_y = val_y.to(self.device)
        with torch.no_grad():
            pred_y = self.model(val_x)
            loss = self.criterion(pred_y, val_y)
        return {'eval_loss': loss.item()}
    

class MNISTDiffTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion,
        train_data,
        eval_data=None,
        lr=1.e-3,
        output_dir='./trainer_output',
        exp_name=None,
        distributed=False,
        ):

        super(MNISTDiffTrainer, self).__init__(
            model, 
            criterion, 
            train_data,
            eval_data,
            lr,
            output_dir,
            exp_name,
            distributed,
        )
        self.diffusion = Diffusion()


    def training_step(self, batch):

        imgs, labels = batch
        
        imgs = imgs.to(self.device)
        t = np.random.randint(0, self.diffusion.num_steps, imgs.shape[0], dtype=int)
        x_t, eps = self.diffusion.sample_xt(imgs, t)
        epsilon_pred = self.model(x_t, torch.from_numpy(t).to(self.device))
        loss = self.criterion(epsilon_pred, eps)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'train_loss': loss.item()}    

    def evaluation_step(self, batch):
        imgs, labels = batch
        
        imgs = imgs.to(self.device)
        t = np.random.randint(0, self.diffusion.num_steps, imgs.shape[0], dtype=int)
        x_t, eps = self.diffusion.sample_xt(imgs, t)
        epsilon_pred = self.model(x_t, torch.from_numpy(t).to(self.device))
        loss = self.criterion(epsilon_pred, eps)
        return {'eval_loss': loss.item()}
    
    def evaluate(self, current_step, eval_dataloader):
        self.model.eval()
        if eval_dataloader is not None:
            eval_losses = []
            for batch in eval_dataloader:
                out = self.evaluation_step(batch)
                eval_losses.append(out['eval_loss'])
            
            if self.distributed:
                # aggregate losses across all devices
                losses_tensor = torch.tensor(eval_losses, device=self.device)
                dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
                mean_loss = torch.mean(losses_tensor).item()/dist.get_world_size()

            else:
                mean_loss = np.mean(eval_losses)

            if self.is_main_process():
                logger.info(f"Rank: {dist.get_rank()} Mean loss across all: {mean_loss}")
                self.writer.add_scalar(
                    "Loss/eval",
                    mean_loss,
                    current_step
                    )
        if self.is_main_process():
            with torch.no_grad():
                x = torch.randn(16, 1, 32,32).to(self.device)
                sample = self.diffusion.inverse(self.model, x=x)
                grid = torchvision.utils.make_grid(sample, nrow=4) 
                self.writer.add_image('generated_images', grid, current_step)
        self.model.train()
