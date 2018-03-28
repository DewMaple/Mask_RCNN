import colorsys
import glob
import time

import cv2
import numpy as np
import os
import random

import coco
import model as modellib
import utils

OUTPUT_DIR = "output"


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = []
    for h, s, v in hsv:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((b, g, r))
    random.shuffle(colors)
    return colors


def fname(image_file):
    return image_file.split(os.sep)[-1]


def save_result(image, result, output_file):
    boxes = result['rois']
    class_ids = result['class_ids']
    # scores = result['scores']

    num_instances = boxes.shape[0]
    num_person = 0
    colors = random_colors(num_instances)
    for i in range(num_instances):
        color = colors[i]
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        class_id = class_ids[i]
        if class_id != 1:
            continue
        num_person += 1
        y1, x1, y2, x2 = boxes[i]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(image, str(num_person), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imwrite(output_file, image)


def images_in_dir(image_dir):
    filenames = []
    for ext in ('*.png', '*.gif', '*.jpg', '*.jpeg'):
        filenames.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(filenames)


def batch_detect(image_dir):
    image_files = images_in_dir(image_dir)

    for imf in image_files:
        image = cv2.imread(imf)
        start = time.time()
        # Run detection
        results = model.detect([image], verbose=1)
        print("time {:.2f} secs for {}".format(time.time() - start, imf))

        # Visualize results
        r = results[0]
        output_file = os.path.join(OUTPUT_DIR, fname(imf))
        save_result(image, r, output_file)


config = InferenceConfig()
config.display()

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = "/Users/administrator/Documents/video/out/images"

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person']

# Load a random image from the images folder
# file_names = next(os.walk(IMAGE_DIR))[2]

# visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

batch_detect(IMAGE_DIR)
