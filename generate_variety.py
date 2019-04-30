import os, cv2, sys, shutil, subprocess, json
sys.path.append("face_segmentation")
sys.path.append("modified_SPADE")
import numpy as np
from face_segmentation.test import face_seg_api
from multiprocessing import Pool, Manager
from modified_SPADE.util.util import tile_images
import cognitive_face as CF
from face import get_facial_attributes
KEYS = ['2b914fc564954fbd81f44a834507c941', '3e7944d9f00a4b03ba9fc5364afffcd8','00cddd78f5c64e29ba0d755a842aaa41']  # Replace with a valid subscription key (keeping the quotes in place).
current_key=0
CF.Key.set(KEYS[current_key])

BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/'  # Replace with your regional Base URL
CF.BaseUrl.set(BASE_URL)



def duplicate_images(images_dir, num_copies):
	file_list = list(os.listdir("temp/labels/"))
	file_list.sort()
	for file in file_list:
		with Pool() as pool:
			pool.starmap(shutil.copyfile, [(images_dir+"/"+file, "temp/images/"+file.replace(".png","_{:03d}.png".format(i))) for i in range(num_copies)])
			pool.starmap(shutil.copyfile, [("temp/labels/"+file, "temp/labels/"+file.replace(".png","_{:03d}.png".format(i))) for i in range(num_copies)])
		os.remove("temp/labels/"+file)

def create_json(images_dir):
	with open("dataset/ffhq/metadata.json",'r') as fh:
		attrib_dict=json.load(fh)
	with open("temp/metadata.json",'w') as fh:
		meta_dict={}
		file_list = list(os.listdir(images_dir))
		file_list.sort()
		for file in file_list:
			counter = 0
			try:
				attrib_dict[file]
			except:
				attrib_dict=get_facial_attributes(images_dir+"/"+file, attrib_dict, KEYS)
			for gender in ['male', 'female']:
				for age in range(0,100,10):
					for glasses in ['NoGlasses', 'ReadingGlasses']:
						fname = file.replace(".png","_{}.png".format(counter))
						meta_dict[fname]={
										'faceAttributes':{
														'smile':smile,
														'gender':gender,
														'age':age,
														'glasses':glasses,
														'emotion':list(attrib_dict[file]['faceAttributes']['emotion'].values())
														}
						}
						counter+=1
		json.dump(meta_dict, fh)
	return counter


def create_json_age(images_dir):
	with open("dataset/ffhq/metadata.json",'r') as fh:
		attrib_dict=json.load(fh)
	with open("temp/metadata.json",'w') as fh:
		meta_dict={}
		file_list = list(os.listdir(images_dir))
		file_list.sort()
		for file in file_list:
			counter = 0
			try:
				attrib_dict[file]
			except:
				attrib_dict=get_facial_attributes(images_dir+"/"+file, attrib_dict, KEYS)
			for age in range(0,100,10):
				fname = file.replace(".png","_{:03d}.png".format(counter))
				meta_dict[fname] = {
	'faceAttributes':{
					'smile':attrib_dict[file]['faceAttributes']['smile'],
					'gender':attrib_dict[file]['faceAttributes']['gender'],
					'age':age,
					'glasses':attrib_dict[file]['faceAttributes']['glasses'],
					'emotion':list(attrib_dict[file]['faceAttributes']['emotion'].values())
					}
			}
				counter+=1
		json.dump(meta_dict, fh)
	return counter

def create_json_gender(image_dir):
	with open("dataset/ffhq/metadata.json",'r') as fh:
		attrib_dict=json.load(fh)
	with open("temp/metadata.json",'w') as fh:
		meta_dict={}
		file_list = list(os.listdir(images_dir))
		file_list.sort()
		for file in file_list:
			counter = 0
			try:
				attrib_dict[file]
			except:
				attrib_dict=get_facial_attributes(images_dir+"/"+file, attrib_dict, KEYS)
			for gender in ['male', 'female']:
				for glasses in ['NoGlasses', 'ReadingGlasses']:
					fname = file.replace(".png","_{:03d}.png".format(counter))
					meta_dict[fname] = {
	'faceAttributes':{
					'smile':attrib_dict[file]['faceAttributes']['smile'],
					'gender':gender,
					'age':attrib_dict[file]['faceAttributes']['age'],
					'glasses':glasses,
					'emotion':list(attrib_dict[file]['faceAttributes']['emotion'].values())
					}
			}
					counter+=1
		json.dump(meta_dict, fh)
	return counter

def move_images_into_dirs(num_copies, results_dir):
	with open("temp/metadata.json",'r') as fh:
		js=json.load(fh)
	rt=int(num_copies**0.5)
	file_list = list(os.listdir(results_dir))
	file_list.sort()
	d={}
	for file in file_list:
		root = file[:file.index("_")]+".png"
		try:
			d[root].append(file)
		except:
			d[root]=[file]
	for file in d.keys():
		fdir = "temp/categ/"+file[:file.index(".png")]
		os.mkdir(fdir)
		
		with Pool() as pool:
			pool.starmap(add_text,[(results_dir+f,"Age:{}".format(js[f]['faceAttributes']['age']),fdir+"/"+f) for f in d[file]])
			#pool.starmap(add_text,[(results_dir+f,"".format(js[f]['faceAttributes']['age']),fdir+"/"+f) for f in d[file]])
		
		subprocess.call(["python", "modified_SPADE/util/tile_images.py", "--grid-size={}".format(rt), "--save-dir=temp/{}".format(file), "--root-dir={}".format(fdir)])

def add_text(image_in, text, image_out):
	image=cv2.imread(image_in)
	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,115)
	fontScale              = 0.4
	fontColor              = (0,0,255)
	lineType               = 2
	cv2.putText(image, text, 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    1,
    lineType)
	cv2.imwrite(image_out, image)

def create_dirs(list_dirs):
	with Pool() as pool:
		pool.map(os.mkdir, list_dirs)
def clean_dirs(list_dirs):
	with Pool() as pool:
		pool.map(shutil.rmtree, list_dirs)

if __name__=="__main__":
	images_dir=sys.argv[1]
	create_dirs(["temp/", "temp/images/", "temp/labels/", "temp/synth/", "temp/tiled/", "temp/categ/"])

	subprocess.call(["python", "face_segmentation/test.py", "--data-folder={}".format(images_dir), "--shape-predictor=face_segmentation/weights/shape_predictor_68_face_landmarks.dat",
																  "--pre-trained=face_segmentation/checkpoints/model.pt", "--save-dir=temp/labels/"])
	num_copies = create_json_age(images_dir)
	duplicate_images(images_dir, num_copies)
	subprocess.call(["python", "modified_SPADE/test.py", 
					"--name=ffhq128", "--dataset_mode=emotion", "--label_dir=temp/labels/".format(images_dir), "--image_dir=temp/images/".format(images_dir), 
					"--emotion=temp/metadata.json".format(images_dir), "--label_nc=8", "--use_vae", "--no_instance", "--emo_dim=14", "--crop_size=128",
					"--z_dim=128", "--load_size=128", "--num_upsampling_layers=less", "--batchSize=12", "--results_dir=temp/synth/"])
	
	move_images_into_dirs(num_copies, "temp/synth/ffhq128/test_latest/images/synthesized_image/")
	#clean_dirs(["temp/images/", "temp/labels/", "temp/synth/", "temp/tiled/", "temp/categ/"])
