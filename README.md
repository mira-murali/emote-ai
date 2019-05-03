# emote-ai
Changing facial emotions using GANs.

To run this you are going to need face segmentation net and SPADE net. Installation instructions inside the respective directories (`face_segmentation` and `modified_SPADE`).

```
python generate_variety.py path/to/directory/with/test/images/
```

This script takes a set of testing images and uses the face API to extract metadata and then generates images for all possible ages from 0-100. Saves generated images, tiled images and GIFs under ./temp/
