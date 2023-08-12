import os
import shutil

from PIL import Image
from tqdm import tqdm

import torch.nn.functional as F
from torchvision import transforms as TF

DATASET_PATH = "./data/raw/Case_Study_1/Raw_Images"
GCP_IMG_PATH = "./data/gcp/Case_Study_1/CGP_Images"


def contains_cgp(img_path):
    """Check if an image contains CGP.

    Args:
        img (str): Image name.

    Returns:
        bool: True if image contains CGP, False otherwise.
    """
    K = 5
    img = Image.open(img_path)
    img = TF.PILToTensor()(img).unsqueeze(0).float()
    img = F.interpolate(img, size=(750, 1000), mode="bilinear", align_corners=False)
    folds = F.unfold(img, kernel_size=K, stride=1, padding=1)
    folds = folds.view(3, K, K, -1)
    whites = folds > 200
    contains = whites.all(dim=0).all(dim=0).all(dim=0).any().item()
    return contains


def find_cgp_and_move(path, outpath):
    """Find CGP files in a directory.

    Args:
        path (str): Path to directory.

    Returns:
        list: List of CGP files.
    """
    imgs = os.listdir(path)
    for img in tqdm(imgs):
        if contains_cgp(f"{path}/{img}"):
            shutil.copy(f"{path}/{img}", f"{outpath}/{img}")
            

def gcp_finder():
    path = DATASET_PATH
    out = GCP_IMG_PATH
    os.makedirs(out, exist_ok=True)
    print("Finding CGP images...")
    cpgs = find_cgp_and_move(path, out)
    
            
gcp_dict =  {
    1: (1, [[571, 575], [516, 520], [459], [636, 640]]),
    2: (1, [[582, 586], [610, 624], [494, 506]]),
    3: (2, [[344, 350], [408], [291, 295], [235, 244]]),
    4: (2, [[305, 312], [320, 331], [416, 417]]),
    5: (2, [[803, 805], [780, 783], [749, 752]]),
    6: (1, [[850, 856]]),
    7: (1, [[253, 258], [75, 91], [160, 181], [286, 297], [357, 361]]),
    8: (1, [[33, 36], [56, 60], [116, 120]])
}

prefix = {1: "DJI_", 2: "2_DJI_"}


def gcp_mover():
     inpath = DATASET_PATH
     out = GCP_IMG_PATH
     
     for point, (num, imgs) in gcp_dict.items():
        os.makedirs(f"{out}/{point}", exist_ok=True)
        for img in imgs:
            img_range = range(img[0], img[-1] + 1) if len(img) > 1 else img
            for i in img_range:
                img_name = f"{prefix[num]}{str(i).zfill(4)}.JPG"
                print(f"Moving {img_name} to {point}")
                shutil.copy(f"{inpath}/{img_name}", f"{out}/{point}/{img_name}")
 


if __name__ == "__main__":
    gcp_mover()
