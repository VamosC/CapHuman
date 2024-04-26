from share import *
from libs.utils import get_args
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader
from src.datasets import create_dataset
from src.cldm.model import create_model, load_state_dict
from einops import rearrange
import argparse
import os
import torch.multiprocessing
from ldm.modules.attention import MemoryEfficientCrossAttention, CrossAttention


if __name__ == '__main__':
    args = get_args()
    print(args)

    # Configs
    resume_path = args.resume_path
    batch_size = args.batch_size
    learning_rate = args.lr

    sd_locked = True if args.sd_locked != 0 else False
    sd_encoder_locked = True if args.sd_encoder_locked != 0 else False
    only_mid_control = False

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(args.config, args).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.unet_lr = learning_rate if args.unet_lr is None else args.unet_lr
    model.sd_locked = sd_locked
    model.sd_encoder_locked = sd_encoder_locked
    model.model.diffusion_model.sd_encoder_locked = sd_encoder_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataset, collate_fn = create_dataset(**vars(args))
    print(f'num of datapoints: {len(dataset)}')

    dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    ckpter = ModelCheckpoint(every_n_train_steps=args.save_interval)
    strategy = DDPStrategy()
    trainer = pl.Trainer(gpus=args.num_gpus, strategy=strategy, precision=args.precision, default_root_dir=args.root, callbacks=[ckpter])

    # Train!
    trainer.fit(model, dataloader)
