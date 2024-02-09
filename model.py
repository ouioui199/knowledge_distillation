from argparse import Namespace
from abc import ABC, abstractmethod
from pathlib import Path

import lightning as L

import torch
from torch import nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss, Linear
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from torchmetrics.classification import Accuracy

from torchvision.models import efficientnet_b0, squeezenet1_0, mobilenet_v3_small


class BaseModel(L.LightningModule, ABC):
    def __init__(self, opt: Namespace):
        super().__init__()
        self.opt = opt
        self.lr = opt.start_lr
        
        self.ce_loss = CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=10)

    @abstractmethod
    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        
        return {'loss': self.ce_loss(logits, labels), 'metrics': self.accuracy(logits, labels)}
    
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self.log('step_loss', outputs['loss'], prog_bar=True)
        self.log('step_metrics', outputs['metrics'])
        
        if not self.train_step_outputs:
            self.train_step_outputs = {
                'step_loss': [outputs['loss']],
                'step_metrics': [outputs['metrics']]
            }
        else:
            self.train_step_outputs['step_loss'].append(outputs['loss'])
            self.train_step_outputs['step_metrics'].append(outputs['metrics'])
    
    def on_train_epoch_end(self):
        tb_logger = self.loggers[0].experiment
        
        mean_loss_value = torch.tensor(self.train_step_outputs['step_loss']).mean()
        mean_metrics_value = torch.tensor(self.train_step_outputs['step_metrics']).mean()
        
        tb_logger.add_scalar('Loss/loss', mean_loss_value, self.current_epoch)
        tb_logger.add_scalar('Metrics/accuracy', mean_metrics_value, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        
        return {'val_loss': self.ce_loss(logits, labels), 'metrics': self.accuracy(logits, labels)}
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        self.log('step_val_loss', outputs['val_loss'], prog_bar=True)
        self.log('step_val_metrics', outputs['metrics'])
        
        if not self.valid_step_outputs:
            self.valid_step_outputs = {
                'step_val_loss': [outputs['val_loss']],
                'step_val_metrics': [outputs['metrics']]
            }
        else:
            self.valid_step_outputs['step_val_loss'].append(outputs['val_loss'])
            self.valid_step_outputs['step_val_metrics'].append(outputs['metrics'])
    
    def on_validation_epoch_end(self) -> None:
        tb_logger = self.loggers[1].experiment
        
        mean_loss_value = torch.tensor(self.valid_step_outputs['step_val_loss']).mean()
        mean_metrics_value = torch.tensor(self.valid_step_outputs['step_val_metrics']).mean()
        
        self.log('val_loss', mean_loss_value)
        self.log('val_Accuracy', mean_metrics_value)
        
        tb_logger.add_scalar('Loss/loss', mean_loss_value, self.current_epoch)
        tb_logger.add_scalar('Metrics/accuracy', mean_metrics_value, self.current_epoch)

    def configure_optimizers(self):
        optimizer = Adam(params=self.parameters(), lr=self.lr)
        optim_dict = {'optimizer': optimizer}
        if self.opt.lr_plateau:
            optim_dict['lr_scheduler'] = {
                'scheduler':ReduceLROnPlateau(
                    optimizer, 
                    patience=self.opt.scheduler_patience, 
                    verbose=True, 
                    threshold=0.005,
                    threshold_mode='abs'
                    ),
                'monitor': 'val_loss'
            }
        else:
            lambda1 = lambda epoch: self.opt.rate_decay ** epoch
            optim_dict['lr_scheduler'] = LambdaLR(optimizer, lr_lambda=lambda1, verbose=True)
            
        return optim_dict


class BigModel(BaseModel):
    def __init__(self, opt: Namespace):
        super().__init__(opt)

        self.model = efficientnet_b0(weights='IMAGENET1K_V1')
        self.model.classifier[1] = Linear(1280, 10)
        
        self.train_step_outputs = {}
        self.valid_step_outputs = {}
        
    def forward(self, x):
        return self.model(x)
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.train_step_outputs.clear()
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self.valid_step_outputs.clear()


class SmallModel(BaseModel):
    def __init__(self, opt: Namespace, num_classes=10):
        super().__init__(opt)
        self.features = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # nn.Conv2d(16, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=8),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, 320),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(320, num_classes)
        )
        
        self.model = nn.Sequential(
            self.features,
            nn.Flatten(1),
            self.classifier
        )
        
        self.train_step_outputs = {}
        self.valid_step_outputs = {}
        
        self.initialize_network(self.model)
    
    @staticmethod
    def initialize_network(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.features(x)
        # flattened_conv_output = torch.flatten(x, 1)
        # x = self.classifier(flattened_conv_output)
        # # flattened_conv_output_after_pooling = torch.nn.functional.avg_pool1d(flattened_conv_output, 2)
        logits = self.model(x)
        flattened_conv_output = torch.flatten(self.features(x), 1)
        return logits, flattened_conv_output
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        
        return {'loss': self.ce_loss(logits[0], labels), 'metrics': self.accuracy(logits[0], labels)}
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self.train_step_outputs.clear()
        
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        
        return {'val_loss': self.ce_loss(logits[0], labels), 'metrics': self.accuracy(logits[0], labels)}
        
    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        self.valid_step_outputs.clear()


class KDModel(BaseModel):
    def __init__(self, opt: Namespace):
        super().__init__(opt)
        self.opt = opt
        
        teacher_ckpt_path = Path('weights_storage/version_1/epoch=49-step=39100.ckpt')
        self.teacher = BigModel.load_from_checkpoint(teacher_ckpt_path, opt=opt, map_location=self.device)
        self.teacher.eval()
        
        self.student = SmallModel(opt)
        
        self.ce_loss = CrossEntropyLoss()
        self.cos_emb_loss = CosineEmbeddingLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        
        self.lr = self.opt.start_lr
        
        self.train_step_outputs = {}
        self.valid_step_outputs = {}

    def forward(self, x):
        with torch.no_grad():
            out_teacher = self.teacher.model.features(x)
            out_teacher = self.teacher.model.avgpool(out_teacher).squeeze()
            
        out_student = self.student(x)
        return {'teacher': out_teacher, 'student': out_student}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        
        assert outputs['teacher'].shape == outputs['student'][1].shape, 'teacher output and student flattened features have different shapes'
        
        emb_loss = self.cos_emb_loss(outputs['teacher'], outputs['student'][1], target=torch.ones(images.size(0), device='mps'))
        logits_loss = self.ce_loss(outputs['student'][0], labels)
        
        return {
            'loss': emb_loss + logits_loss, 
            'emb_loss': emb_loss,
            'logits_loss': logits_loss,
            'metrics': self.accuracy(outputs['student'][0], labels)
        }
    
    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        log_dict = {'step_loss': outputs['loss']}
        log_dict.update({'step_' + k: v for k, v in outputs.items() if k != 'loss'})
        self.log_dict(log_dict)
        
        if not self.train_step_outputs:
            self.train_step_outputs = {k: [v] for k, v in log_dict.items()}
        else:
            for k in log_dict.keys():
                self.train_step_outputs[k].append(log_dict[k])
    
    def on_train_epoch_end(self):
        tb_logger = self.loggers[0].experiment
        
        mean_metrics_value = torch.tensor(self.train_step_outputs['step_metrics']).mean()
        tb_logger.add_scalar('Metrics/accuracy', mean_metrics_value, self.current_epoch)
        
        mean_losses_values = {k: torch.tensor(v).mean() for k, v in self.train_step_outputs.items() if 'loss' in k}
        for k, v in mean_losses_values.items():
            tb_logger.add_scalar('Loss/' + k.replace('step_', ''), v, self.current_epoch)
        
        self.train_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        
        assert outputs['teacher'].shape == outputs['student'][1].shape, 'teacher output and student flattened features have different shapes'
        
        emb_loss = self.cos_emb_loss(outputs['teacher'], outputs['student'][1], target=torch.ones(images.size(0), device='mps'))
        logits_loss = self.ce_loss(outputs['student'][0], labels)
        
        return {
            'val_loss': emb_loss + logits_loss, 
            'emb_loss': emb_loss,
            'logits_loss': logits_loss, 
            'metrics': self.accuracy(outputs['student'][0], labels)
        }
    
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx = 0) -> None:
        log_dict = {'step_val_loss': outputs['val_loss']}
        log_dict.update({'step_val_' + k: v for k, v in outputs.items() if k != 'loss'})
        self.log_dict(log_dict)
        
        if not self.valid_step_outputs:
            self.valid_step_outputs = {k: [v] for k, v in log_dict.items()}
        else:
            for k in log_dict.keys():
                self.valid_step_outputs[k].append(log_dict[k])
    
    def on_validation_epoch_end(self) -> None:
        tb_logger = self.loggers[1].experiment
        
        mean_accuracy_value = torch.tensor(self.valid_step_outputs['step_val_metrics']).mean()
        tb_logger.add_scalar('Metrics/accuracy', mean_accuracy_value, self.current_epoch)
        
        mean_losses_values = {k: torch.tensor(v).mean() for k,v in self.valid_step_outputs.items() if 'loss' in k}
        for k,v in mean_losses_values.items():
            tb_logger.add_scalar('Loss/' + k.replace('step_val_', ''), v, self.current_epoch)
            
        self.log('val_loss', mean_losses_values['step_val_loss'])
        self.log('val_Accuracy', mean_accuracy_value)
        
        
        self.valid_step_outputs.clear()