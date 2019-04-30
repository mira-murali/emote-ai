import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def tile_image():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default='./',
                        help='folder containing images')
    parser.add_argument('--grid-size', type=int, default=3,
                        help='size of grid')
    parser.add_argument('--img-size', type=int, default=128,
                        help='size of image')
    parser.add_argument('--save-dir', type=str, default='./',
                        help='save grid image')
    args = parser.parse_args()
    images = os.listdir(args.root_dir)
    grid_image = np.zeros((args.grid_size*args.img_size, args.grid_size*args.img_size, 3), dtype=np.uint8)

    img_counter = 0
    for i in range(args.grid_size):
        for j in range(args.grid_size):
            if img_counter >= len(images):
                break
            img = Image.open(os.path.join(args.root_dir, images[img_counter]))
            img = np.asarray(img, dtype=np.uint8)
            grid_image[i*args.img_size:(i+1)*args.img_size, j*args.img_size:(j+1)*args.img_size, :] = img
            img_counter += 1

    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(args.save_dir, grid_image)

if __name__ =='__main__':
    tile_image()
