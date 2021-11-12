import argparse
import torch
import pytorch_lightning as pl

from mask_contrast import MaskContrast
from data.dataloaders.dataset import DatasetKeyQuery
from utils.config import create_config, load_config
from utils.common_config import get_train_dataset, get_train_transformations,\
                                get_train_dataloader, get_optimizer, adjust_learning_rate
from utils.collate import collate_custom
from pytorch_lightning.callbacks import ModelSummary


parser = argparse.ArgumentParser(description='Main function')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
args = parser.parse_args()


def main():
    pl.seed_everything(42)
    
    p = load_config(args.config_exp)
    # Dataset
    train_transform = get_train_transformations()
    print(train_transform)
    train_dataset = DatasetKeyQuery(get_train_dataset(p, transform=None), train_transform, 
                                downsample_sal=not p['model_kwargs']['upsample'])
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=p['train_batch_size'], shuffle=True,
                    num_workers=p['num_workers'], pin_memory=True, drop_last=True, collate_fn=collate_custom)

    # Lightning module
    model = MaskContrast(p)

    trainer = pl.Trainer(
        max_epochs=p['epochs'],
        gpus=p['gpus'],
        tpu_cores=p['tpu_cores'] if p['tpu_cores'] > 0 else None,
        # sync_batchnorm=True if p['gpus'] > 1 or p['tpu_cores'] > 1 else False,
        precision=32 if p['fp32'] else 'bf16',
        # limit_train_batches=1,
        log_every_n_steps=25,
        fast_dev_run=p['fast_dev_run'],
        default_root_dir=p['default_root_dir']
    )
    trainer.fit(model, train_dataloaders=train_dataloader)


if __name__ == '__main__':
    main()
