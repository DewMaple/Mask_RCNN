import os
import time

import cv2
import face_recognition as fr
import numpy as np
from img_utils.files import images_in_dir, filename

import coco
import model as modellib
import utils

OUTPUT_DIR = "output_front_face"


def save_result(image, result, output_file):
    boxes = result['rois']
    class_ids = result['class_ids']
    # scores = result['scores']

    num_instances = boxes.shape[0]
    num_person = 0
    for i in range(num_instances):
        if not np.any(boxes[i]):
            continue
        class_id = class_ids[i]
        if class_id != 1:
            continue
        num_person += 1
        y1, x1, y2, x2 = boxes[i]
        roi = image[y1:y2, x1:x2]

        faces = fr.face_locations(roi, number_of_times_to_upsample=4, model='cnn')
        if len(faces) > 0:
            top, right, bottom, left = faces[0]
            print(top, right, bottom, left)
            cv2.rectangle(image, (left + x1, top + y1), (right + x1, bottom + y1), (0, 0, 255), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(image, str(num_person), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imwrite(output_file, image)


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
        output_file = os.path.join(OUTPUT_DIR, filename(imf))
        save_result(image, r, output_file)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

IMAGE_DIR = "/Users/administrator/Documents/video/out/images"
# IMAGE_DIR = "/Users/administrator/workspace/Mask_RCNN/samples/person/coco/val"

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person']
batch_detect(IMAGE_DIR)
