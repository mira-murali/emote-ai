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
from imutils import face_utils
import dlib

np.set_printoptions(threshold=sys.maxsize)
# load pre-trained model and weights
MAP_FEATURES = {'mouth': 0, 
                'reyeb': 1, 
                'leyeb': 2, 
                'reye': 3,
                'leye': 4,
                'nose': 5 
                }
COLOR_FEATURES = {'mouth': 4, 
                'reyeb': 5, 
                'leyeb': 5, 
                'reye': 6,
                'leye': 6,
                'nose': 7 
                }

def load_model():
    model = MobileNetV2_unet(None).to(args.device)
    state_dict = torch.load(args.pre_trained)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def landmarking(gray):
    rois = []
    # detect faces in the grayscale image
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

	# loop over the face parts individually
        for (_, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            rois.append((x, y, w, h))
    print("Done")
    return rois

def assign_color(image, roi):
    for key, value in MAP_FEATURES.items():
        x, y, w, h = roi[value]
        image[y:y+h, x:x+w] = COLOR_FEATURES[key]
    print("Done color")
    return image

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
    parser.add_argument("-p", "--shape-predictor", required=True,
	                    help="path to facial landmark predictor")
    args = parser.parse_args()
    args.device = torch.device('cuda:1' if torch.cuda.is_available else 'cpu')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    
    test_dataset = TestDataset(path=args.data_folder, transform=transform, save_path=args.save_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = load_model()
   
    for i, (gray_images, images, names) in enumerate(test_loader):
        # Forward Pass
        gray_images = gray_images.numpy()
        with Pool() as p:
            rois = p.starmap(landmarking, zip(gray_images))
    
        images = images.to(args.device)
        logits = model(images)
        mask = np.argmax(logits.data.cpu().numpy(), axis=1)
        with Pool() as p:
            mask = p.starmap(assign_color, zip(mask, rois))
        mask = np.asarray(mask) 
        norm_mask = (mask - mask.min())/(mask.max() - mask.min())
        norm_mask *= 255.0
        # Save
        with Pool() as p:
            p.starmap(cv2.imwrite, zip(names, norm_mask))
