import os, cv2, sys, shutil, subprocess
sys.path.append("face_segmentation")
sys.path.append("modified_SPADE")
import numpy as np
from face_segmentation.test import face_seg_api
from multiprocessing import Pool, Manager


def duplicate_images(images_dir, num_copies):
	file_list = list(os.listdir("temp/labels/"))
	file_list.sort()
	for file in file_list:
		with Pool() as pool:
			pool.starmap(shutil.copyfile, [(images_dir+"/"+file, "temp/images/"+file.replace(".png","_{}.png".format(i))) for i in range(num_copies)])
			pool.starmap(shutil.copyfile, [("temp/labels/"+file, "temp/labels/"+file.replace(".png","_{}.png".format(i))) for i in range(num_copies)])
		os.remove("temp/labels/"+file)

def create_json(images_dir):
	attrib_dict={
	'faceAttributes':{
					'smile':0.0,
					'gender':'male',
					'age':0,
					'glasses':'NoGlasses',
					'emotion':{
							'anger': 0.0,
							'contempt': 0.0,
							'disgust': 0.0,
							'fear': 0.0,
							'happiness': 0.0,
							'neutral': 0.0,
							'sadness': 0.0,
							'surprise': 0.0
							}

					}
			}
	with open("temp/metadata.json",'w') as fh:
		meta_dict={}
		file_list = list(os.listdir(images_dir))
		file_list.sort()
		for file in file_list:
			counter = 0
			for smile in [0, 0.5, 1.0]:
				for gender in ['male', 'female']:
					for age in [0,25,50,75]:
						for glasses in ['NoGlasses', 'ReadingGlasses', 'Sunglasses', 'SwimmingGoggles']:
							for emotion in ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']:
								fname = file.replace(".png","_{}.png".format(counter))
								meta_dict[fname] = attrib_dict
								meta_dict[fname]={
												'faceAttributes':{
																'smile':smile,
																'gender':'gender',
																'age':age,
																'glasses':glasses,
																'emotion':{
																		'anger': 0.0,
																		'contempt': 0.0,
																		'disgust': 0.0,
																		'fear': 0.0,
																		'happiness': 0.0,
																		'neutral': 0.0,
																		'sadness': 0.0,
																		'surprise': 0.0
																		}

																}
								}
								meta_dict[fname]['faceAttributes']['emotion'][emotion]=1.0
								counter+=1
		json.dump(meta_dict, fh)
	return counter

if __name__=="__main__":
	images_dir=sys.argv[1]
	os.mkdir("temp/")
	os.mkdir("temp/images/")
	os.mkdir("temp/labels/")

	face_seg_api(images_dir, segmentation_predictor="face_segmentation/checkpoints/model.pt",
					 landmark_predictor="face_segmentation/weights/shape_predictor_68_face_landmarks.dat")		#Puts the labels in images_dir/temp/labels/
	num_copies = create_json(images_dir)
	duplicate_images(images_dir, num_copies)
	subprocess.call(["python", "modified_SPADE/test.py", 
					"--name ffhq128", "--dataset_mode emotion", "--label_dir {}/temp/labels/".format(images_dir), "--image_dir {}/temp/images/".format(images_dir), 
					"--emotion {}/temp/metadata.json".format(images_dir), "--label_nc 8", "--use_vae", "--no_instance", "--emo_dim 14", "--crop_size 128",
					"--z_dim 128", "--load_size 128", "--num_upsampling_layers less", "--batchSize 12"])