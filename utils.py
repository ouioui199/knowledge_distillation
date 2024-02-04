from argparse import ArgumentParser

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.utilities import rank_zero_only

class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
    
    
class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        metrics = {k: v for k, v in metrics.items() if ('step' not in k) and ('val' not in k)}
        return super().log_metrics(metrics, step)
    
    
def train_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--version', type=int, required=True)
    
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--val_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=64)
    
    parser.add_argument('--start_lr', type=float, default=0.001)
    parser.add_argument('--lr_plateau', action='store_true')
    parser.add_argument('--scheduler_patience', type=int, default=20)
    parser.add_argument('--rate_decay', type=float, default=0.998)
    
    parser.add_argument('--early_stopping', type=int, default=30)
    
    return parser