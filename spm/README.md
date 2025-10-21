---
license: other
tags:
- vision
- image-matching
inference: false
pipeline_tag: keypoint-detection
---


# SuperPoint

## Overview

The SuperPoint model was proposed
in [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629) by Daniel
DeTone, Tomasz Malisiewicz and Andrew Rabinovich.

This model is the result of a self-supervised training of a fully-convolutional network for interest point detection and
description. The model is able to detect interest points that are repeatable under homographic transformations and
provide a descriptor for each point. The use of the model in its own is limited, but it can be used as a feature
extractor for other tasks such as homography estimation, image matching, etc.

The abstract from the paper is the following:

*This paper presents a self-supervised framework for training interest point detectors and descriptors suitable for a
large number of multiple-view geometry problems in computer vision. As opposed to patch-based neural networks, our
fully-convolutional model operates on full-sized images and jointly computes pixel-level interest point locations and
associated descriptors in one forward pass. We introduce Homographic Adaptation, a multi-scale, multi-homography
approach for boosting interest point detection repeatability and performing cross-domain adaptation (e.g.,
synthetic-to-real). Our model, when trained on the MS-COCO generic image dataset using Homographic Adaptation, is able
to repeatedly detect a much richer set of interest points than the initial pre-adapted deep model and any other
traditional corner detector. The final system gives rise to state-of-the-art homography estimation results on HPatches
when compared to LIFT, SIFT and ORB.*

## Demo notebook

A demo notebook showcasing inference + visualization with SuperPoint can be found [here](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SuperPoint/Inference_with_SuperPoint_to_detect_interest_points_in_an_image.ipynb).

## How to use

Here is a quick example of using the model to detect interest points in an image:

```python
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(image, return_tensors="pt")
outputs = model(**inputs)
```

The outputs contain the list of keypoint coordinates with their respective score and description (a 256-long vector).

You can also feed multiple images to the model. Due to the nature of SuperPoint, to output a dynamic number of keypoints,
you will need to use the mask attribute to retrieve the respective information :

```python
from transformers import AutoImageProcessor, SuperPointForKeypointDetection
import torch
from PIL import Image
import requests

url_image_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_1 = Image.open(requests.get(url_image_1, stream=True).raw)
url_image_2 = "http://images.cocodataset.org/test-stuff2017/000000000568.jpg"
image_2 = Image.open(requests.get(url_image_2, stream=True).raw)

images = [image_1, image_2]

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

inputs = processor(images, return_tensors="pt")
outputs = model(**inputs)
```

We can now visualize the keypoints.

```python
import matplotlib.pyplot as plt
import torch

for i in range(len(images)):
    image = images[i]
    image_width, image_height = image.size  

    image_mask = outputs.mask[i]
    image_indices = torch.nonzero(image_mask).squeeze()

    image_scores = outputs.scores[i][image_indices]
    image_keypoints = outputs.keypoints[i][image_indices]
    
    keypoints = image_keypoints.detach().numpy()
    scores = image_scores.detach().numpy()

    valid_keypoints = [
        (kp, score) for kp, score in zip(keypoints, scores)
        if 0 <= kp[0] < image_width and 0 <= kp[1] < image_height
    ]

    valid_keypoints, valid_scores = zip(*valid_keypoints)
    valid_keypoints = torch.tensor(valid_keypoints)
    valid_scores = torch.tensor(valid_scores)

    print(valid_keypoints.shape)

    plt.axis('off')
    plt.imshow(image)
    plt.scatter(
        valid_keypoints[:, 0], 
        valid_keypoints[:, 1], 
        s=valid_scores * 100, 
        c='red'
    )
    plt.show()
```

This model was contributed by [stevenbucaille](https://huggingface.co/stevenbucaille).
The original code can be found [here](https://github.com/magicleap/SuperPointPretrainedNetwork).

```bibtex
@inproceedings{detone2018superpoint,
  title={Superpoint: Self-supervised interest point detection and description},
  author={DeTone, Daniel and Malisiewicz, Tomasz and Rabinovich, Andrew},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition workshops},
  pages={224--236},
  year={2018}
}
```