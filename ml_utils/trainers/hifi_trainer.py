import os
import tqdm
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod

import torch
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import get_rank, is_initialized
import torch.distributed as dist

import logging
import itertools

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trainer_log.log") 
    ]
)

logger = logging.getLogger(__name__)

class HiFiTrainer():
    def __init__(
        self,
        generator,
        mpd,
        msd,
        train_data,
        eval_data=None,
        # criterion=None,
        lr=1.e-3,
        beta1=0.9,
        beta2=0.999,
        hifi_gan_config=None,
        output_dir='./trainer_output',
        exp_name=None,
        distributed=False,
        data_collator=None
        ):

        self.distributed = distributed
        if distributed:
            self.device = self.init_process_group()
            self.generator = generator.to(self.device)
            self.mpd = mpd.to(self.device)
            self.msd = msd.to(self.device)
            self.generator = DDP(
                self.generator, device_ids=[self.device], 
                find_unused_parameters=True
                )
            self.mpd = DDP(
                self.mpd, device_ids=[self.device], 
                find_unused_parameters=True
                )
            self.msd = DDP(
                self.msd, device_ids=[self.device], 
                find_unused_parameters=True
                )
            self.num_devices = dist.get_world_size()
            if not self.is_main_process():
                logging.disable()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
            self.generator = generator.to(self.device)
            self.mpd = mpd.to(self.device)
            self.msd = msd.to(self.device)
            self.num_devices = 1
        
        self.train_data = train_data
        self.eval_data = eval_data
        self.data_collator = data_collator
        self.optim_g, self.optim_d = self.configure_optimizers(lr, beta1, beta2)
        self.scheduler_g = None
        self.scheduler_d = None
        self.hifi_gan_config = hifi_gan_config
        self.lr_decay = hifi_gan_config.lr_decay
        
        self.output_dir = output_dir
        logger.info(f"Training results saved to: \n {self.output_dir}")
        
        # making sure sub-directories are created in output_dir
        
        if exp_name is None:
            exp_name = 'exp-111'
        self.exp_name = exp_name
        # self.exp_name = self.get_exp_name(exp_name)
        self.log_dir = os.path.join(output_dir, 'logs')
        self.chkpt_dir = os.path.join(output_dir, 'checkpoints', self.exp_name)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.chkpt_dir, exist_ok=True)   
        
        if self.is_main_process():
            # summary writer for creating logs
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.exp_name))

        self.gradient_accumulation_steps = 1
        self.train_dataloader = None
        self.eval_dataloader = None


    # def forward_step(self, batch):
    #     """Implement forward pass for the model and return loss"""
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
    #     return loss

    
    # def training_step(self, batch, step):
    #     loss = self.forward_step(batch)
    #     loss.backward()

    #     if (step+1) % self.gradient_accumulation_steps == 0 or (step+1) == len(self.train_dataloader):
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()

    #     return {'train_loss': loss.item()}    

    # def evaluation_step(self, batch):
    #     with torch.no_grad():
    #         loss = self.forward_step(batch)
    #     return {'eval_loss': loss.item()}
    

    # def evaluate(self, current_step):
    #     if self.eval_dataloader is not None:
    #         self.model.eval()
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
    #         self.model.train()
    
    def configure_optimizers(self, lr, beta1, beta2):
        optim_g = torch.optim.AdamW(self.generator.parameters(), lr, betas=[beta1, beta2])
        optim_d = torch.optim.AdamW(itertools.chain(self.msd.parameters(), self.mpd.parameters()),
                                lr, betas=[beta1, beta2])

        return optim_g, optim_d


    def get_chkpt_name(self, chkpt_name=None):
        """Returns valid checkpoint name if name is entered,
        otherwise returns the name of latest checkpoint for the current exp_name.
        """
        if chkpt_name is None:
            chkpt_names = []
            for ent in os.scandir(self.chkpt_dir):
                if '.pth' in ent.name:
                    chkpt_names.append(ent.name)

            if len(chkpt_names)==0:
                # not checkpoint found...
                return None

            substr = '-steps-'
            st_indices = [name.find(substr) + len(substr) for name in chkpt_names]
            num_steps = [int(name[idx:idx+7]) for name, idx in zip(chkpt_names, st_indices)]
            latest_chkpt_idx = np.argmax(num_steps)
            chkpt_name = chkpt_names[latest_chkpt_idx]
        elif '.pth' not in chkpt_name:
            chkpt_name += '.pth'
        return chkpt_name


    def get_train_dataloader(self, batch_size):
        """Returns train dataloaders"""
        if self.distributed:
            train_dataloader_kwargs = {
                'shuffle': False,
                'sampler': DistributedSampler(
                                self.train_data, shuffle=True,
                                num_replicas=dist.get_world_size(), rank=dist.get_rank()
                                )}
        else:
            train_dataloader_kwargs = {'shuffle': True}

        if self.data_collator is not None:
            train_dataloader_kwargs['collate_fn'] = self.data_collator
            
        train_dataloader = torch.utils.data.DataLoader(
            self.train_data, batch_size=batch_size, **train_dataloader_kwargs
        )
        return train_dataloader

    def get_eval_dataloader(self, batch_size):
        """Returns eval dataloaders"""
        if self.eval_data is not None:
            if self.distributed:
                eval_dataloader_kwargs = {
                    'shuffle': False,
                    'sampler': DistributedSampler(
                        self.eval_data, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())}
            else:
                eval_dataloader_kwargs = {'shuffle': False}

            if self.data_collator is not None:
                eval_dataloader_kwargs['collate_fn'] = self.data_collator
        
            eval_dataloader = torch.utils.data.DataLoader(
                self.eval_data, batch_size=batch_size, **eval_dataloader_kwargs
            )
        else:
            eval_dataloader = None
        return eval_dataloader


    def train(self,
            batch_size, 
            batch_size_per_device=None,
            max_steps=None,
            num_epochs=None,
            resume_training=False,
            chkpt_save_steps=None,
            start_from_chkpt=None,  
            train_log_steps=None, 
            eval_log_steps=None, 
        ):

        assert batch_size % self.num_devices == 0, f"batch_size must be multiple of num_devices: {self.num_devices}"
        if batch_size_per_device is None:
            batch_size_per_device = batch_size//self.num_devices
            self.gradient_accumulation_steps = 1
        else:
            assert batch_size % batch_size_per_device == 0, f"batch_size must be multiple of 'batch_size_per_device': {batch_size_per_device}"
            self.gradient_accumulation_steps = batch_size//(batch_size_per_device*self.num_devices)
        
        self.train_dataloader = self.get_train_dataloader(batch_size_per_device)
        self.eval_dataloader = self.get_eval_dataloader(batch_size_per_device)

        num_batches = len(self.train_dataloader)
        
        assert max_steps is not None or num_epochs is not None, f"At least provide one 'max_steps' or 'num_epochs'"
        if max_steps is None:
            max_steps = num_epochs*num_batches//self.gradient_accumulation_steps

        if train_log_steps is None:
            train_log_steps = num_batches//10
        if eval_log_steps is None:
            eval_log_steps = num_batches//2

        current_step = 0
        if resume_training:
            chkpt_name = self.get_chkpt_name(start_from_chkpt)
            if chkpt_name:
                current_step =  self.load_checkpoint(chkpt_name)
                logger.info(f"Training from checkpoint: '{chkpt_name}'")
            else:
                logger.warning(f"Checkpoint not found, training from scratch.")

        current_epoch = current_step//(num_batches//self.gradient_accumulation_steps)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=self.lr_decay, last_epoch=current_epoch)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=self.lr_decay, last_epoch=current_epoch)

        start_time = datetime.now()
        logger.info(f"Training starts at {start_time}")
        logger.info(f"Steps per epoch: {num_batches//self.gradient_accumulation_steps}")
        logger.info(f"Total steps: {max_steps}")
        logger.info(f"Starting from: {current_step}")
        logger.info(f"Training loss logged every: {train_log_steps} steps")
        logger.info(f"Runs 'Evaluate'  every: {eval_log_steps} steps")

        logger.info(f"Batch size per device: {batch_size_per_device}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {batch_size}")


        if chkpt_save_steps:
            if self.eval_dataloader is not None:
                assert chkpt_save_steps%eval_log_steps ==0, f"'chkpt_save_steps' must be integer multiple of 'eval_log_steps'"
            logger.info(f"Checkpoint saved every: {chkpt_save_steps} steps")
        
        self.model.train()
        if self.is_main_process():
            pbar = tqdm.tqdm(total=max_steps - current_step, desc='Training progress', unit='step')            
            pbar.set_postfix(step=current_step, loss="N/A")
            # pbar.update(1)
        # current_step += 1
        train_losses = []
        mean_loss = 'N/A'
        epoch=0
        h = self.hifi_gan_config

        self.generator.train()
        self.mpd.train()
        self.msd.train()
        while (current_step < max_steps):
            if self.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)
            self.optimizer.zero_grad()
            for itr, batch in enumerate(self.train_dataloader):

                x, y, y_mel = batch
                y = y.unsqueeze(1)

                y_g_hat = self.generator(x)
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size,
                    h.win_size, h.fmin, h.fmax_for_loss)

                self.optim_d.zero_grad()()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                self.optim_d.step()

                # Generator
                self.optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                loss_gen_all.backward()
                self.optim_g.step()


                # train_out = self.training_step(batch, itr)
                train_losses.append(loss_gen_all.item())

                if current_step % eval_log_steps ==0 and \
                    ((itr+1)%self.gradient_accumulation_steps == 0 or (itr+1) == len(self.train_dataloader)):
                    # extra condition to avoid evaluating multiple times in case of gradient accumulation

                    # Always calls this method, make sure to 
                    # check for eval_dataloader is None within evaluate
                    with torch.no_grad():
                        if self.eval_dataloader is not None and self.distributed:
                            self.eval_dataloader.sampler.set_epoch(epoch)
                        self.evaluate(current_step)

                if current_step % train_log_steps ==0 and \
                    ((itr+1)%self.gradient_accumulation_steps == 0 or (itr+1) == len(self.train_dataloader)):
                    # extra condition to avoid evaluating multiple times in case of gradient accumulation
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    if self.distributed:
                        # aggregate losses across all devices
                        losses_tensor = torch.tensor(train_losses, device=self.device)
                        dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
                        mean_loss = torch.mean(losses_tensor).item()/dist.get_world_size()
                    else:
                        mean_loss = np.mean(train_losses)
                    train_losses = []

                    # logging training loss...
                    if self.is_main_process():
                        pbar.set_postfix(step=current_step, loss=mean_loss)
                        self.writer.add_scalar(
                            "gen_loss_total/train",
                            mean_loss,
                            current_step
                            )
                        self.writer.add_scalar(
                            "mel_spec_error/train",
                            mel_error,
                            current_step
                            )
                
                # checkpoint saving and progress bar update..
                if self.is_main_process() and \
                    ((itr+1)%self.gradient_accumulation_steps == 0 or (itr+1) == len(self.train_dataloader)):
                    # extra condition to avoid evaluating multiple times in case of gradient accumulation
                    if chkpt_save_steps is not None and current_step % chkpt_save_steps ==0 and current_step !=0:
                        self.save_checkpoint(current_step)
                    # if (itr+1)%self.gradient_accumulation_steps == 0 or (itr+1) == len(self.train_dataloader):
                    pbar.set_postfix(step=current_step, loss=mean_loss)
                    pbar.update(1)

                if (itr+1)%self.gradient_accumulation_steps == 0 or (itr+1) == len(self.train_dataloader):
                    current_step += 1
                if current_step >= max_steps:
                    break
            epoch += 1
            self.scheduler_g.step()
            self.scheduler_d.step()

        if self.is_main_process():
            pbar.close()
            self.writer.close()

            self.save_checkpoint(current_step)
            logger.info(f"Model trained for {current_step} steps & results saved at"+\
                         f"\n\n checkpoint: {self.chkpt_dir} \n logs: {self.log_dir}"+\
                        f"\n exp-name: {self.exp_name}\n")

            end_now = datetime.now()
            logger.info(f"Training ends at {end_now}")
            logger.info(f"Training duration: {end_now-start_time}")

        if self.distributed:
            dist.destroy_process_group()

    def save_checkpoint(self, step):
        checkpoint = {
            "generator_state_dict": self.generator.module.state_dict() if hasattr(self.generator, "module") else self.generator.state_dict(),
            "mpd_state_dict": self.mpd.module.state_dict() if hasattr(self.mpd, "module") else self.mpd.state_dict(),
            "msd_state_dict": self.msd.module.state_dict() if hasattr(self.msd, "module") else self.msd.state_dict(),
            
            "optim_g_state_dict": self.optim_g.state_dict(),
            "optim_d_state_dict": self.optim_d.state_dict(),
            "scheduler_g_state_dict": self.scheduler_g.state_dict() if self.scheduler_g else None,
            "scheduler_d_state_dict": self.scheduler_d.state_dict() if self.scheduler_d else None,
            "step": step,
            "rng_state": torch.get_rng_state(),
            # "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        checkpoint_path = os.path.join(self.chkpt_dir, f"checkpoint-steps-{step:07d}.pth")
        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, chkpt_name):
        checkpoint_path = os.path.join(self.chkpt_dir, chkpt_name)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        generator_state_dict = checkpoint["generator_state_dict"]
        mpd_state_dict = checkpoint["mpd_state_dict"]
        msd_state_dict = checkpoint["msd_state_dict"]

        if self.distributed:
            # adjust for DDP wrapped module
            from collections import OrderedDict
            self.generator.load_state_dict(
                OrderedDict({'module.'+k:v for k,v in generator_state_dict.items()})
                )
            self.mpd.load_state_dict(
                OrderedDict({'module.'+k:v for k,v in mpd_state_dict.items()})
                )
            self.msd.load_state_dict(
                OrderedDict({'module.'+k:v for k,v in msd_state_dict.items()})
                )
        else:
            self.generator.load_state_dict(generator_state_dict)
            self.mpd.load_state_dict(mpd_state_dict)
            self.msd.load_state_dict(msd_state_dict)

        self.optim_g.load_state_dict(checkpoint["optim_g_state_dict"])
        self.optim_d.load_state_dict(checkpoint["optim_d_state_dict"])

        if self.scheduler_g and checkpoint["scheduler_g_state_dict"] is not None:
            self.scheduler_g.load_state_dict(checkpoint["scheduler_g_state_dict"])
        if self.scheduler_d and checkpoint["scheduler_d_state_dict"] is not None:
            self.scheduler_d.load_state_dict(checkpoint["scheduler_d_state_dict"])
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"].cpu())
        # if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
        #     torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        return checkpoint["step"]  # Returning the last saved step


    @staticmethod
    def is_main_process():
        return not dist.is_initialized() or dist.get_rank() == 0

    @staticmethod
    def init_process_group():
        """Initlizes distributed setup, using environment variables"""

        rank = os.environ.get("SLURM_PROCID", "Not set")
        local_rank = os.environ.get("SLURM_LOCALID", "Not set")
        world_size = os.environ.get("SLURM_NTASKS", "Not set")

        # Get the address of the master node (rank 0)
        master_addr = os.environ.get('MASTER_ADDR', 'Not set')
        master_port = os.environ.get('MASTER_PORT', 'Not set')

        logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        logging.info(f"cuda device count: {torch.cuda.device_count()}")
        logging.info(f"global rank: {rank}")
        logging.info(f"local rank: {local_rank}")
        logging.info(f"world size: {world_size}")
        logging.info(f"master addr: {master_addr}")
        logging.info(f"master port: {master_port}")

        # Device setup
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)

        else:
            device = torch.device("cpu")
            logging.warning(f"device={device} for global rank: {rank}")

        dist.init_process_group(
            backend='nccl',
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=int(rank),
            world_size=int(world_size),
        )
        logging.info(f"Process group successfully initialized...!")
        return device
    

