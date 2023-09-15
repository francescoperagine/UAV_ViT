import os
import torch
import torch.nn.functional as TF
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import v2

class SquarePad:
    def __call__(self, image):
        _, w, h = image.size()
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [vp, vp, hp, hp]
        padded_img = TF.pad(image, padding)
        return padded_img


class PlotsDataset(Dataset):
    def __init__(self, annotations, img_dir, img_size, transforms = None, antialias = True):
        self.img_labels = annotations
        self.img_dir = img_dir
        self.img_size = img_size
        self.antialias = antialias
        
        # Initialize the transform list by adding in the square padding
        compose_square_pad = v2.Compose([ SquarePad() ])
        compose_resize_convert = v2.Compose([
            v2.Resize(size=self.img_size, antialias=self.antialias),
            v2.ToImageTensor(),
            v2.ConvertImageDtype(torch.float32),
            # v2.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])
        # compose_normalization = v2.Compose([v2.Normalize(mean=self.means, std=self.stds)])

        self.transform = compose_square_pad

        # Add in any user-specified transforms
        self.transform = v2.Compose([self.transform, transforms]) if(transforms is not None) else self.transform

        # Add in the rest of the transforms
        self.transform = v2.Compose([self.transform, compose_resize_convert])

        # self.means, self.stds = self.calculate_mean_std()
        # print(f"Means: {self.means}, Stds: {self.stds}")

        # Normalize based on dataset mean and standard deviation
        # self.transform = v2.Compose([self.transform, compose_normalization])

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        row = self.img_labels.iloc[index]

        image_path = os.path.join(os.getcwd(), self.img_dir, row['plot'])
        image = read_image(image_path)
        image = self.transform(image)

        label = row['elev'].astype('float32')

        return image, label

    def save_dataset(self, filename):
        with open(filename, 'wb') as f:
            torch.save(self, f)