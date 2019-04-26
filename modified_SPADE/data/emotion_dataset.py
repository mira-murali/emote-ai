"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
import json, sys, os, torch
from PIL import Image
import numpy as np
from data.base_dataset import BaseDataset, get_params, get_transform

class EmotionDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        parser.add_argument('--emotion', type=str, required=True, default='',
                            help='path to the directory that contains emotions.')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        with open(opt.emotion,'r') as fh:
            self.emotions = json.load(fh)
        return label_paths, image_paths, instance_paths

    def postprocess(self, input_dict, only_emotion=False):
        if only_emotion:
            metadata = self.emotions[os.path.basename(input_dict['path'])]
            temp_emo = torch.zeros(self.opt.emo_dim, dtype=torch.float)
            temp_emo[metadata] = 1
            input_dict['meta'] = temp_emo
            return
        attributes = self.emotions[os.path.basename(input_dict['path'])]['faceAttributes']
        smile = attributes['smile']
        age = attributes['age']/100
        gender = 1 if attributes['gender']=='male' else 0
        emotion = list(attributes['emotion'].values())
        glasses = {'NoGlasses':[0,0,0], 'ReadingGlasses':[0,0,1], 'Sunglasses':[0,1,0], 'SwimmingGoggles':[1,0,0]}
        spectacles = glasses[attributes['glasses']]
        metadata = emotion + [smile, age, gender] + spectacles
        metadata = torch.FloatTensor(metadata).cuda()
        input_dict['meta'] = metadata
        return

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        
        #Loki
        #label_tensor = transform_label(label) * 255.0
        label_tensor = transform_label(label)*7.1
        label_tensor = label_tensor.type(torch.LongTensor)

        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }
        
        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict, only_emotion=True)

        return input_dict
