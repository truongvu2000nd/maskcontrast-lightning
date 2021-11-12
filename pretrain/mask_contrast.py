import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torchmetrics import Accuracy
from torch.optim.lr_scheduler import LambdaLR
from modules.moco.builder import ContrastiveModel

from utils.utils import freeze_layers
from utils.common_config import get_optimizer


class MaskContrast(pl.LightningModule):
    def __init__(self, p) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = ContrastiveModel(p)
        self.acc1 = Accuracy()
        self.acc5 = Accuracy(top_k=5)

        if p['freeze_layers']:
            freeze_layers(self.model)

    def training_step(self, batch, batch_idx):
        # Forward pass
        # im_q = batch['query']['image']
        # im_k = batch['key']['image']
        # sal_q = batch['query']['sal']
        # sal_k = batch['key']['sal']
        im_q, im_k, sal_q, sal_k = batch

        logits, labels, saliency_loss = self.model(im_q=im_q, im_k=im_k, sal_q=sal_q, sal_k=sal_k)

        # Use E-Net weighting for calculating the pixel-wise loss.
        uniq, freq = torch.unique(labels, return_counts=True)
        p_class = torch.zeros(logits.shape[1], dtype=torch.float32, device=self.device)
        p_class_non_zero_classes = freq.float() / labels.numel()
        p_class[uniq] = p_class_non_zero_classes
        w_class = 1 / torch.log(1.02 + p_class)
        contrastive_loss = F.cross_entropy(logits, labels, weight=w_class,
                                            reduction='mean')

        # Calculate total loss and update meters
        loss = contrastive_loss + saliency_loss

        self.log("contrastive_loss", contrastive_loss, prog_bar=True)
        self.log("saliency_losses", saliency_loss, prog_bar=True)
        self.log("losses", loss, prog_bar=True)

        # contrastive_losses.update(contrastive_loss.item())
        # saliency_losses.update(saliency_loss.item())
        # losses.update(loss.item())

        acc1, acc5 = self.acc1(logits, labels), self.acc5(logits, labels)
        self.log("acc1", acc1, prog_bar=True)
        self.log("acc5", acc5, prog_bar=True)

        # acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        # top1.update(acc1[0], im_q.size(0))
        # top5.update(acc5[0], im_q.size(0))
        return loss

    def configure_optimizers(self):
        p = self.hparams.p
        optimizer = get_optimizer(p, self.model.parameters())

        # poly scheduler
        scheduler = {
            "scheduler": LambdaLR(optimizer, lambda epoch: pow(1-(epoch/p['epochs']), 0.9)),
            "interval": "epoch",
        } 

        return [optimizer], [scheduler]