from typing import Any, Dict, Tuple
import time
import torch
import wandb
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import BinaryAccuracy, MulticlassF1Score
import numpy as np
from typing import Union
import os
import pandas as pd
import json
class FAMLITrainingModule(LightningModule):
    """
    A PyTorch Lightning module for training a classification model.

    This module encapsulates a neural network, loss function, optimizers, learning rate schedulers, 
    and evaluation metrics for streamlined training, validation, and testing workflows.

    References:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        output_dir: str,
        num_classes: int = 2,
        loss_weights: Union[int, list] = 3.0,
        use_head_only: bool = False
        ):
        """
        Initializes the training module.

        :param model: The neural network model to train.
        :param optimizer: Optimizer for training.
        :param scheduler: Learning rate scheduler.
        :param compile: Flag to enable model compilation (for PyTorch 2.0+).
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.save_dir = output_dir
        self.model = net.cuda()
        self.compile = compile
        # Define the loss function
        self.num_classes = num_classes
        self.only_head = use_head_only

        
        if num_classes == 2:
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_weights))
            self.train_accuracy = BinaryAccuracy()
            self.val_accuracy = BinaryAccuracy()
            self.test_accuracy = BinaryAccuracy()
        else:
            
            self.criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weights))

            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            
            self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            
            
        self.train_f1 = MulticlassF1Score(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes)

        # Metrics for tracking loss
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.test_loss_metric = MeanMetric()

        # Track the best validation accuracy
        self.best_val_accuracy = MaxMetric()
        self.best_val_f1 = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        :param x: Input tensor.
        :return: Output logits.
        """
        return self.model(x)

    def on_train_start(self) -> None:
        """
        Hook executed at the start of training.
        Resets validation metrics to ensure no carryover from sanity checks.
        """
        self.val_loss_metric.reset()
        self.val_accuracy.reset()
        self.val_f1.reset()
        self.best_val_f1.reset()
        self.best_val_accuracy.reset()

        

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a single forward step on a batch of data.

        :param batch: Dictionary containing input data and target labels.
            - 'stacked_img': The input tensor.
            - 'us_lie': The target labels.
        :return: Tuple containing:
            - Loss tensor.
            - Predictions tensor.d
            - Target labels tensor.
        """
        start = time.time()
        inputs, targets = batch["img"].to(self.device), batch["gt"].to(self.device)
        if self.num_classes==2:
            targets=targets.unsqueeze(dim=-1).float()
        # if self.num_classes==2:
        #     targets = targets.unsqueeze(dim=-1).float()
        logits, _ = self.forward(inputs)
        
        loss = self.criterion(logits, targets)
        
        # predictions = torch.argmax(logits, dim=1)
        if self.num_classes==2:
            predictions = torch.tensor((logits>0)).float()
        else:
            predictions = torch.argmax(logits, dim=1)

        end = time.time()
        return loss, predictions, targets, logits

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Executes a training step.

        :param batch: Batch of training data.
        :param batch_idx: Index of the current batch.
        :return: Training loss tensor.
        """
        loss, predictions, targets, logits = self.model_step(batch)

        # Update metrics
        self.train_loss_metric(loss)
        self.train_accuracy(predictions, targets)
        self.train_f1(predictions, targets)
        
        if self.global_step % 100 == 0:
            self.logger.experiment.log({
            "train/logits": wandb.Histogram(predictions.detach().cpu().numpy().astype(np.float32)),
            "global_step": self.global_step
        })
        
        
        # Log metrics
        self.log("train/loss", self.train_loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Executes a validation step.

        :param batch: Batch of validation data.
        :param batch_idx: Index of the current batch.
        """
        loss, predictions, targets, logits = self.model_step(batch)
        # Update metrics
        self.val_loss_metric(loss)
        print(predictions.shape, targets.shape, predictions, targets)
        self.val_accuracy(predictions, targets)
        self.val_f1 (predictions, targets)  
        
        
        self.logger.experiment.log({
            "val/logits": wandb.Histogram(predictions.detach().cpu().numpy().astype(np.float32)),
            "global_step": self.global_step
        })  

        # Log metrics
        self.log("val/loss", self.val_loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Hook executed at the end of a validation epoch.
        Updates and logs the best validation accuracy.
        """
        current_val_accuracy = self.val_accuracy.compute()
        current_val_f1 = self.val_f1.compute()
        
        self.best_val_accuracy(current_val_accuracy)
        self.best_val_f1(current_val_f1)
        self.log("val/best_accuracy", self.best_val_accuracy.compute(), sync_dist=True, prog_bar=True)
        self.log("val/best_f1", self.best_val_f1.compute(), sync_dist=True, prog_bar=True)

    def on_test_start(self) -> None:
        self.save_dir = os.path.join(self.save_dir, "predictions")
        os.makedirs(self.save_dir, exist_ok=True)
        self.test_preds = []  # Dict of domain -> list of prediction dicts
    
    
    def on_test_end(self) -> None:
        
        df = pd.DataFrame(self.test_preds)
        save_path = os.path.join(self.save_dir, f"predictions.csv")
        df.to_csv(save_path, index=False)
        print(f"[✓] Saved predictions to {save_path}")
    

    def test_step(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Executes a test step.

        :param batch: Batch of test data.
        :param batch_idx: Index of the current batch.
        """
        loss, predictions, targets, logits = self.model_step(batch)

        # Update metrics
        self.test_loss_metric(loss)
        self.test_accuracy(predictions, targets)
        self.test_f1(predictions, targets)

        study_ids = batch.get("study_id_x", ["unknown"] * predictions.shape[0])
        file_paths = batch.get("fname", ["unknown"] * predictions.shape[0])

        # for sid, path, pred, gt, logit in zip(
        #     study_ids, file_paths, predictions, targets, logits
        # ):
        #     self.test_preds.append({
        #         "study_id": sid,
        #         "file_path": path,
        #         "logit": logit.item(),
        #         "output": pred.item(),
        #         "gt": gt.item(),
        #     })

        # Log metrics
                
        self.logger.experiment.log({
            "test/logits": wandb.Histogram(predictions.detach().cpu().numpy().astype(np.float32)),
            "global_step": self.global_step
        })
        self.log("test/loss", self.test_loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

    
    def on_predict_start(self) -> None:
        self.save_dir = os.path.join(self.save_dir, "predictions")
        os.makedirs(self.save_dir,exist_ok=True)
        if self.only_head:
            cols = ["embed","logit","output"]
        else:
            cols = ["study_id", "file_path", "logit", "output", "gt", "embed"]
        self.pred_df = pd.DataFrame(columns=cols)
    
    def on_predict_end(self) -> None:
        save_csv = f"{self.save_dir}/preds_and_embed.csv"
        self.pred_df.to_csv(save_csv, index=False)
        print(f"Predictions saved to: {save_csv}")
        
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Executes a test step.

        :param batch: Batch of test data.
        :param batch_idx: Index of the current batch.
        """
                
        inputs, targets = batch["img"].to(self.device), batch["gt"].to(self.device)
        if self.num_classes==2:
            targets=targets.unsqueeze(dim=-1).float()
        # if self.num_classes==2:
        #     targets = targets.unsqueeze(dim=-1).float()
        logits_and_embed = self.forward(inputs)
        
        logits, embed = logits_and_embed
        
                # predictions = torch.argmax(logits, dim=1)
        if self.num_classes==2:
            predictions = torch.tensor((logits>0)).float()
        else:
            predictions = torch.argmax(logits, dim=1)
        
        if not isinstance(logits_and_embed, tuple):
            raise ValueError("Model should return embedding along with logits to extract the embeddings.")
        
        
        embed_np = embed.cpu().numpy()  # shape: (batch_size, embedding_dim)
        new_data = {
            "study_id": batch["study_id_x"],
            "file_path": batch["fname"],
            "logit": logits.squeeze().cpu().numpy(),
            "output": predictions.squeeze().cpu().numpy(),
            "gt": batch["gt"].cpu().numpy(),
            "embed": [json.dumps(e.tolist()) for e in embed_np]  # list of serialized embeddings
        }

        self.pred_df = pd.concat([self.pred_df, pd.DataFrame(new_data)], ignore_index=True)        

    def setup(self, stage: str) -> None:
        """
        Hook executed during setup stages like training or testing.
        Can dynamically adjust or compile the model.

        :param stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        if self.compile and stage == "fit":
            self.model = torch.compile(self.model)
    
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler:
            scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                    mode='min',
                                                                    factor=0.1,
                                                                    patience=5,
                                                                    threshold=1e-4,
                                                                    verbose=True),
            'monitor': 'val/f1',  # <-- must match validation metric name
            'interval': 'epoch',
            'frequency': 1
        }
    
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    
