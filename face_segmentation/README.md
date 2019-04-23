# Face Segmentation
Semantic segmentation for hair, face and background

Barebones version of [this repository](https://github.com/akirasosa/mobile-semantic-segmentation/).

## Dataset
Labeled faces in the wild. Get it from [here](https://www.dropbox.com/s/kkj73eklp5fnut0/data.zip?dl=0).


## Training
```
python train.py \
    --data-folder data/lfw \
    --pre-trained weights/mobilenet_v2.pth.tar \
    --output-folder scratch/ \
    --num-epochs=50 \
    --batch-size=32
```

## Facial Landmarking (Dependencies)

You need to have Dlib and imutils installed. You can install them using pip. Make sure you also have OpenCV installed:

```
pip install dlib
pip install imutils
```

You also need to download the `shape_predictor_68_face_landmarks.dat` from [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
## Testing

```
python test.py --data-folder data/samples/ --pre-trained checkpoints/model.pt --shape-predictor shape_predictor_68_face_landmarks.dat --save-dir seg_results/
```

The `save-dir` folder will be created if it doesn't exist to store the segmentation results.


## Results
![alt text](results.png "Sample results")
