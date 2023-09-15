import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def plot_loss(train_loss, val_loss):
        # Plot training and validation loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_predictions(actual, predicted):
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(actual, predicted)
        plt.plot([min(actual), max(actual)], [min(actual), max(actual)], color='red', linestyle='--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.show()