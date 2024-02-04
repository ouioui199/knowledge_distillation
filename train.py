from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import v2

import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from model import KDModel, BigModel, SmallModel
from utils import TBLogger, CustomProgressBar, train_parser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = train_parser(parser)
    opt = parser.parse_args()

    transform = v2.Compose([
        # v2.GaussianBlur(kernel_size=(3, 3)),
        # v2.RandomAutocontrast(),
        # v2.RandomAdjustSharpness(sharpness_factor=1.5),
        # v2.RandomEqualize(),
        v2.RandomRotation(90),
        v2.RandomHorizontalFlip(p=0.7),
        v2.RandomVerticalFlip(p=0.7),
        # v2.RandomPerspective(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # v2.Normalize((0.1307,), (0.3081,))
    ])

    # train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
    # val_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    val_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=opt.batch_size, 
        shuffle=True, 
        num_workers=opt.workers, 
        persistent_workers=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=opt.val_batch_size, 
        shuffle=True, 
        num_workers=opt.val_workers, 
        persistent_workers=True,
        pin_memory=True
    )

    model = KDModel(opt)
    trainer = L.Trainer(
        max_epochs=opt.epochs, 
        num_sanity_val_steps=0,
        benchmark=True,
        callbacks=[
            CustomProgressBar(),
            EarlyStopping(
                monitor='val_loss', 
                verbose=True,
                patience=opt.early_stopping,
                min_delta=0.005
            ),
            LearningRateMonitor(logging_interval='epoch'),
            ModelCheckpoint(
                dirpath='weights_storage/version_' + str(opt.version),
                monitor='val_Accuracy', 
                verbose=True, 
                mode='max'
            )
        ],
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        logger=[
            TBLogger('training_logs', name=None, version=opt.version, sub_dir='train'),
            TBLogger('training_logs', name=None, version=opt.version, sub_dir='valid')
        ]
    )

    # tuner = Tuner(trainer)
    # lr_finder = tuner.lr_find(model, train_dataloaders=train_loader)
    # model.hparams.lr = lr_finder.suggestion()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)