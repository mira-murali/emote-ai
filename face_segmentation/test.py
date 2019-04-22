from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet
import os
from dataset import TestDataset
from multiprocessing import Pool
import sys
np.set_printoptions(threshold=sys.maxsize)
# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to(args.device)
    state_dict = torch.load(args.pre_trained)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Arguments
    parser.add_argument('--data-folder', type=str, default='./data',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='batch size (default: 8)')
    parser.add_argument('--pre-trained', type=str, default=None,
                        help='path of pre-trained weights (default: None)')
    parser.add_argument('--save-dir', type=str, default='seg_results',
                        help='Directory to save results (will be created)')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    
    test_dataset = TestDataset(path=args.data_folder, transform=transform, save_path=args.save_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = load_model()
   
    for i, (images, names) in enumerate(test_loader):
        # Forward Pass
        images = images.to(args.device)
        logits = model(images)
        mask = np.argmax(logits.data.cpu().numpy(), axis=1)
        norm_mask = (mask - mask.min())/(mask.max() - mask.min())
        norm_mask *= 255.0
        # Save
        with Pool() as p:
            p.starmap(cv2.imwrite, zip(names, norm_mask))
