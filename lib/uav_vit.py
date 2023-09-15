import torch.optim as optim
import torch.nn as nn
import torch
import pytorch_lightning as pl
import mlflow.pytorch
import mlflow
from torch.nn import Sequential
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class UAV_vit(pl.LightningModule):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.model.heads = Sequential(nn.Linear(in_features=1024, out_features=1)).float()
        self.loss_fn = loss_fn

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    # Training
    def on_train_start(self) -> None:
        return super().on_train_start()

    def training_step(self, batch, batch_idx):
        outputs, labels = self.format_output(batch)
        result = self.log_step_results("train", outputs, labels)
        self.training_step_outputs.append(result)
        return result

    def on_train_epoch_end(self):
        self.log_epoch_results("train", self.training_step_outputs)
        self.training_step_outputs.clear()

    # Validation

    def validation_step(self, batch, batch_idx):
        outputs, labels = self.format_output(batch)
        result = self.log_step_results("val", outputs, labels)
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        self.log_epoch_results("val", self.validation_step_outputs)
        self.validation_step_outputs.clear()
    
    # Testing

    def test_step(self, batch, batch_idx):
        outputs, labels = self.format_output(batch)
        result = self.log_step_results("test", outputs, labels)
        self.test_step_outputs.append(result)
        return result
    
    def on_test_epoch_end(self):
        self.log_epoch_results("test", self.test_step_outputs)
        self.test_step_outputs.clear()

    # Helper function

    def format_output(self, batch):
        images, labels = batch
        labels = labels.view(-1, 1)
        outputs = self(images)
        return outputs, labels

    # Logging

    def log_step_results(self, prefix, outputs, labels):
        loss = self.loss_fn(outputs, labels)

        print(f"outputs shape: {outputs.shape} - labels shape: {labels.shape}")

        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().numpy()

        print(f"outputsCPU shape: {outputs.shape} - labelsCPU shape: {labels.shape}")

        r2 = r2_score(labels, outputs)
        mae = mean_absolute_error(labels, outputs)
        mse = mean_squared_error(labels, outputs)

        return {"loss": loss, "r2": r2, "mae": mae, "mse": mse}

    def log_epoch_results(self, prefix, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_r2 = sum([x["r2"] for x in outputs]) / len(outputs)
        avg_mae = sum([x["mae"] for x in outputs]) / len(outputs)
        avg_mse = sum([x["mse"] for x in outputs]) / len(outputs)

        self.log(f"{prefix}_loss", avg_loss)
        self.log(f"{prefix}_r2", avg_r2)
        self.log(f"{prefix}_mae", avg_mae)
        self.log(f"{prefix}_mse", avg_mse)

        mlflow.log_metric(f"{prefix}_loss", avg_loss.item())
        mlflow.log_metric(f"{prefix}_r2", avg_r2)
        mlflow.log_metric(f"{prefix}_mae", avg_mae)
        mlflow.log_metric(f"{prefix}_mse", avg_mse)