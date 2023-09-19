from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
import mlflow.pytorch
import mlflow
from torch.nn import Sequential
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class UAV_vit(pl.LightningModule):
    def __init__(self, backbone, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = backbone

        num_input_filters = backbone.heads[0].in_features
        num_output_values = 1

        self.model.heads = nn.Linear(in_features=num_input_filters, out_features=num_output_values).float()

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.losses = {'train': [], 'val': [], 'test': []}
        self.r2_scores = {'train': [], 'val': [], 'test': []}
        self.maes = {'train': [], 'val': [], 'test': []}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    
    # Hooks

    # Training

    def training_step(self, batch, batch_idx):
        phase = "train"
        outputs, labels = self.get_data(batch)
        self.training_step_outputs.append(outputs)
        return self.on_step(phase, outputs, labels)

    def on_train_epoch_end(self):
        self.log_epoch_results("train")
        self.training_step_outputs.clear()

    # Validation

    def validation_step(self, batch, batch_idx):
        phase = "val"
        outputs, labels = self.get_data(batch)
        self.validation_step_outputs.append(outputs)
        return self.on_step(phase, outputs, labels)

    def on_validation_epoch_end(self):
        self.log_epoch_results("val")
        self.validation_step_outputs.clear()
    
    # Testing

    def test_step(self, batch, batch_idx):
        phase = "test"
        outputs, labels = self.get_data(batch)
        self.test_step_outputs.append(outputs)
        return self.on_step(phase, outputs, labels)
    
    def on_test_epoch_end(self):
        self.log_epoch_results("test")
        self.test_step_outputs.clear()

    def on_fit_end(self):
        self.create_scatterplots()

    # Helper functions

    def get_data(self, batch):
        images, labels = batch
        labels = labels.view(-1, 1)
        outputs = self(images)
        return outputs, labels

    def on_step(self, phase, outputs, labels):
        metrics = self.compute_metrics(phase, outputs, labels)
        self.log_step_results(phase, metrics)
        return metrics["loss"]

    def compute_metrics(self, phase, outputs, labels):
        loss = self.loss_fn(outputs, labels)

        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().numpy()        

        r2 = r2_score(labels, outputs)
        mae = mean_absolute_error(labels, outputs)

        return {"loss": loss, "r2": r2, "mae": mae}

    # Logging

    def log_step_results(self, phase, metrics):

        self.losses[phase].append(metrics["loss"].item())
        self.r2_scores[phase].append(metrics["r2"])
        self.maes[phase].append(metrics["mae"])

        self.log("loss", metrics["loss"], on_step=True, on_epoch=True, logger=True)
        self.log("r2", metrics["r2"], on_step=True, on_epoch=True, logger=True)
        self.log("mae", metrics["mae"], on_step=True, on_epoch=True, logger=True)

        mlflow.log_metric(f"{phase}_loss", metrics["loss"].item())
        mlflow.log_metric(f"{phase}_r2", metrics["r2"])
        mlflow.log_metric(f"{phase}_mae", metrics["mae"])
    
    def log_epoch_results(self, prefix):

        avg_loss = sum(self.losses[prefix]) / len(self.losses[prefix])
        avg_r2 = sum(self.r2_scores[prefix]) / len(self.r2_scores[prefix])
        avg_mae = sum(self.maes[prefix]) / len(self.maes[prefix])

        # print("avg_loss", avg_loss)
        # print("avg_r2", avg_r2)
        # print("avg_mae", avg_mae)


        # # avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # # avg_r2 = sum([x["r2"] for x in outputs]) / len(outputs)
        # # avg_mae = sum([x["mae"] for x in outputs]) / len(outputs)

        # self.log(f"{prefix}_loss", avg_loss, on_epoch=True, logger=True)
        # self.log(f"{prefix}_r2", avg_r2, on_epoch=True, logger=True)
        # self.log(f"{prefix}_mae", avg_mae, on_epoch=True, logger=True)

        mlflow.log_metric(f"{prefix}_loss_avg", avg_loss)
        mlflow.log_metric(f"{prefix}_r2_avg", avg_r2)
        mlflow.log_metric(f"{prefix}_mae_avg", avg_mae)

    # Visualization

    def create_scatterplots(self):
        plt.scatter(self.losses['train'], self.r2_scores['train'], label='Training', alpha=0.5)
        plt.scatter(self.losses['val'], self.r2_scores['val'], label='Validation', alpha=0.5)
        plt.scatter(self.losses['test'], self.r2_scores['test'], label='Testing', alpha=0.5)
        plt.xlabel('Loss')
        plt.ylabel('R^2')
        plt.legend()
        plt.title('Scatter Plot: Loss vs R^2')
        plt.show()