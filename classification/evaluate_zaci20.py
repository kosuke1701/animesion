from argparse import ArgumentParser
import os

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_pretrained_vit import ViT
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

def load_image(filename):
    try:
        with open(filename, "rb") as f:
            image = Image.open(f)
            return image.convert("RGB")
    except UserWarning as e:
        print(filename)
        input("Something wrong happens while loading image: {} {}".format(filename, str(e)))

# Example Model definition
class Model(object):
    def __init__(self, ckpt_fn, model_type="L_16", image_size=128):
        state_dict = torch.load(ckpt_fn, map_location="cpu")
        self.model = ViT(
            name=model_type,
            pretrained=True,
            num_classes=1,
            image_size=image_size
        )
        self.model.fc = torch.nn.Identity()
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to("cuda")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                            std=[0.5, 0.5, 0.5])
        ])
    
    # img1, img2: PIL image
    def score(self, img1, img2):
        self.model.eval()

        tensor1, tensor2 = self.transform(img1).unsqueeze(0), self.transform(img2).unsqueeze(0)
        x = torch.cat([tensor1, tensor2], dim=0).to("cuda")

        y = self.model(x)

        score = torch.sum(y[0] * y[1])
        return score.item()
        

if __name__=="__main__":
    parser = ArgumentParser()

    parser.add_argument("--test-pairs", help="CSV file which lists test image pairs.")
    parser.add_argument("--test-dataset-dir", help="Directory of test images.")

    parser.add_argument("--target-fnr", type=float, default=0.139, help="Reference FNR used to compute FPR.")

    args = parser.parse_args()

    model = Model("results/zaci20_l16_size128_batch64_epoch50_decay20_lr1em3_pretrained_2.ckpt")

    df = pd.read_csv(args.test_pairs)
    df = df[df["invalid"]==0]
    true_labels = df["label"].values
    ROOT_DIR = args.test_dataset_dir
    scores = []
    for pathA, pathB, label in tqdm(df[["pathA", "pathB", "label"]].values):
        img1 = load_image(os.path.join(args.test_dataset_dir, pathA))
        img2 = load_image(os.path.join(args.test_dataset_dir, pathB))
        
        score = model.score(img1, img2)
        scores.append(score)
    
    fpr, tpr, threshold = roc_curve(true_labels, scores)
    eer = 1. - brentq(lambda x: 1. - x - interp1d(tpr, fpr)(x), 0., 1.)
    fnr = 1. - tpr
    print("False Positive Rate: ", interp1d(fnr, fpr)(args.target_fnr))
    print("Threshold: ", interp1d(fnr, threshold)(args.target_fnr))
    print("Equal Error Rate: ", eer)