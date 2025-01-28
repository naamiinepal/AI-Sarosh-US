from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


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
    ) -> None:
        """
        Initializes the training module.

        :param model: The neural network model to train.
        :param optimizer: Optimizer for training.
        :param scheduler: Learning rate scheduler.
        :param compile: Flag to enable model compilation (for PyTorch 2.0+).
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = net
        self.compile = compile

        # Define the loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # Metrics for tracking accuracy (both per class and overall)
        self.train_accuracy_per_class = Accuracy(task="multiclass", num_classes=3, average="none")
        self.train_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.val_accuracy_per_class = Accuracy(task="multiclass", num_classes=3, average="none")
        self.val_accuracy = Accuracy(task="multiclass", num_classes=3)
        self.test_accuracy_per_class = Accuracy(task="multiclass", num_classes=3, average="none")
        self.test_accuracy = Accuracy(task="multiclass", num_classes=3)

        # Metrics for tracking loss
        self.train_loss_metric = MeanMetric()
        self.val_loss_metric = MeanMetric()
        self.test_loss_metric = MeanMetric()

        # Track the best validation accuracy
        self.best_val_accuracy = MaxMetric()

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
        self.best_val_accuracy.reset()

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes a single forward step on a batch of data.

        :param batch: Dictionary containing input data and target labels.
            - 'stacked_img': The input tensor.
            - 'us_lie': The target labels.
        :return: Tuple containing:
            - Loss tensor.
            - Predictions tensor.
            - Target labels tensor.
        """
        inputs, targets = batch["stacked_img"], batch["us_lie"]
        logits = self.forward(inputs)
        loss = self.criterion(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        return loss, predictions, targets

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Executes a training step.

        :param batch: Batch of training data.
        :param batch_idx: Index of the current batch.
        :return: Training loss tensor.
        """
        loss, predictions, targets = self.model_step(batch)

        # Update metrics
        self.train_loss_metric(loss)
        self.train_accuracy(predictions, targets)

        # Log metrics
        self.log("train/loss", self.train_loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Executes a validation step.

        :param batch: Batch of validation data.
        :param batch_idx: Index of the current batch.
        """
        loss, predictions, targets = self.model_step(batch)

        # Update metrics
        self.val_loss_metric(loss)
        self.val_accuracy(predictions, targets)

        # Log metrics
        self.log("val/loss", self.val_loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        """
        Hook executed at the end of a validation epoch.
        Updates and logs the best validation accuracy.
        """
        current_val_accuracy = self.val_accuracy.compute()
        self.best_val_accuracy(current_val_accuracy)
        self.log("val/best_accuracy", self.best_val_accuracy.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Executes a test step.

        :param batch: Batch of test data.
        :param batch_idx: Index of the current batch.
        """
        loss, predictions, targets = self.model_step(batch)

        # Update metrics
        self.test_loss_metric(loss)
        self.test_accuracy(predictions, targets)

        # Log metrics
        self.log("test/loss", self.test_loss_metric, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def setup(self, stage: str) -> None:
        """
        Hook executed during setup stages like training or testing.
        Can dynamically adjust or compile the model.

        :param stage: Current stage ('fit', 'validate', 'test', or 'predict').
        """
        if self.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configures the optimizer and scheduler.

        :return: Dictionary containing optimizer and scheduler configurations.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler:
            scheduler = self.hparams.scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = FAMLITrainingModule(model=None, optimizer=None, scheduler=None, compile=False)
