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
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize)
# load pre-trained model and weights
MAP_FEATURES = {'mouth': 0,
                'reyeb': 1,
                'leyeb': 2,
                'reye': 3,
                'leye': 4,
                'nose': 5
                }
COLOR_FEATURES = {'mouth': 3,
                'reyeb': 4,
                'leyeb': 4,
                'reye': 5,
                'leye': 5,
                'nose': 6
                }

def load_model(device, pre_trained):
    model = MobileNetV2_unet(None).to(device)
    state_dict = torch.load(pre_trained)
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
    return rois

def assign_color(image, roi):
    for key, value in MAP_FEATURES.items():
        try:
            x, y, w, h = roi[value]
        except IndexError:
            return np.zeros_like(image)
        image[y:y+h, x:x+w] = COLOR_FEATURES[key]
    return image

def face_seg_api(input_dir, segmentation_predictor="face_segmentation/checkpoints/model.pt",
                     landmark_predictor="face_segmentation/weights/shape_predictor_68_face_landmarks.dat", batch_size=16, device=None):
    import matplotlib.pyplot as plt
    output_dir = "temp/labels/"
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(landmark_predictor)

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    test_dataset = TestDataset(path=input_dir, transform=transform, save_path=output_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = load_model(device, segmentation_predictor)
    namefile = open("temp/empty.txt", 'w')
    for i, (gray_images, images, names) in tqdm(enumerate(test_loader)):
        # Forward Pass
        gray_images, names = gray_images.numpy(), np.asarray(names)
        with Pool() as p:
            rois = p.starmap(landmarking, zip(gray_images))

        images = images.to(device)
        logits = model(images)
        mask = np.argmax(logits.data.cpu().numpy(), axis=1)
        with Pool() as p:
            mask = p.starmap(assign_color, zip(mask, rois))

        mask = np.asarray(mask)
        means = np.mean(np.mean(mask, axis=2), axis=1)
        non_black_images = np.argwhere(means!=0).reshape(-1)
        black_images = np.argwhere(means==0).reshape(-1)
        write_names = names[black_images]
        np.savetxt(namefile, write_names, fmt='%s')
        mask = mask[non_black_images]
        norm_mask = (mask - mask.min())/(mask.max() - mask.min())
        norm_mask *= 255.0
        # Save
        with Pool() as p:
            p.starmap(cv2.imwrite, zip(names[non_black_images], norm_mask))

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Semantic Segmentation')

    # Arguments
    parser.add_argument('--data-folder', type=str, default='../../emote/synthesized_image/',
                        help='name of the data folder (default: ./data)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument('--pre-trained', type=str, default=None,
                        help='path of pre-trained weights (default: None)')
    parser.add_argument('--save-dir', type=str, default='../../emote/si_labels/',
                        help='Directory to save results (will be created)')
    parser.add_argument("-p", "--shape-predictor", required=True,
	                    help="path to facial landmark predictor")
    parser.add_argument('--save-empty', default='temp/empty.txt', type=str,
                        help='path to save txt file with empty image names')
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    print("Loading test images...")
    test_dataset = TestDataset(path=args.data_folder, transform=transform, save_path=args.save_dir)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    model = load_model(args.device, args.pre_trained)
    namefile = open(args.save_empty, 'w')
    for i, (gray_images, images, names) in tqdm(enumerate(test_loader)):
        # Forward Pass
        gray_images, names = gray_images.numpy(), np.asarray(names)
        with Pool() as p:
            rois = p.starmap(landmarking, zip(gray_images))

        images = images.to(args.device)
        logits = model(images)
        mask = np.argmax(logits.data.cpu().numpy(), axis=1)
        with Pool() as p:
            mask = p.starmap(assign_color, zip(mask, rois))

        mask = np.asarray(mask)
        # means = np.mean(np.mean(mask, axis=2), axis=1)
        # non_black_images = np.argwhere(means!=0).reshape(-1)
        # black_images = np.argwhere(means==0).reshape(-1)
        # write_names = names[black_images]
        # np.savetxt(namefile, write_names, fmt='%s')
        # mask = mask[non_black_images]
        norm_mask = (mask - mask.min())/(mask.max() - mask.min())
        norm_mask *= 255.0
        # Save
        with Pool() as p:
            p.starmap(cv2.imwrite, zip(names, norm_mask))
