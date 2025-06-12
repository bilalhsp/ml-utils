import logging
from .base_trainer import BaseTrainer


# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

class SpectDiffTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_data,
        **kwargs,
        ):

        super(SpectDiffTrainer, self).__init__(
            model, 
            train_data,
            **kwargs,
        )


    def forward_step(self, batch):
        audio_spects, labels = batch
        audio_spects = audio_spects.to(self.device)
        # B, C, L = audio_spects.shape

        if hasattr(self.model, "module"):
            loss, xt = self.model.module.compute_loss(audio_spects)
        else:
            loss, xt = self.model.compute_loss(audio_spects)
        return loss



