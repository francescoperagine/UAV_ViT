import matplotlib.pyplot as plt
import numpy as np
import torch

def get_dataset_samples(dataset, name):
    cols, rows = 4, 4
    figure = plt.figure(figsize=(8, 8))
    for i in range(1, cols * rows + 1):
        sample_index = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_index]
        img = img.permute(1,2,0)
        figure.add_subplot(rows, cols, i)
        plt.suptitle("Plot samples " + name)
        filename = dataset.img_labels.iloc[i]['plot'][0:-4]
        plt.title(f'{filename}: {label:.2f}m')        
        plt.axis("off")
        plt.imshow(img)
    plt.show()

def calculate_mean_std(dataset):
    """
    Calculate the mean and standard deviation of the RGB channels for all images in a dataset.

    Args:
        dataset (torch.utils.data.Dataset): The PyTorch dataset containing (image, label) pairs.

    Returns:
        tuple: A tuple containing two lists, one for channel means and one for channel standard deviations.
    """
    channel_sums = [0.0, 0.0, 0.0]
    channel_sums_squared = [0.0, 0.0, 0.0]
    num_samples = len(dataset)

    for img, _ in dataset:
        img = np.array(img)  # Converte il tensore PyTorch in un array NumPy
        for i in range(3):  # Per ogni canale (R, G, B)
            channel_sums[i] += np.sum(img[:, :, i]) # type: ignore
            channel_sums_squared[i] += np.sum(img[:, :, i] ** 2) # type: ignore

    channel_means = [sum / (num_samples * 244 * 244) for sum in channel_sums]
    channel_stds = [np.sqrt((sum_squared / (num_samples * 244 * 244)) - (mean ** 2)) for sum_squared, mean in zip(channel_sums_squared, channel_means)]

    return channel_means, channel_stds