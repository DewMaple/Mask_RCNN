import datetime
import os
import sys

import numpy as np
import skimage.color
import skimage.draw
# Root directory of the project
import skimage.io
from img_utils.files import images_in_dir

import model as modellib
import utils
import visualize
from samples.person.person_conf import PersonConfig
from samples.person.person_dataset import PersonDataset
from samples.person.training import train

ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/person"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH = '/Users/administrator/workspace/AI_models/mask_rcnn/mask_rcnn_coco.h5'


def validate_images(image_dir, subset='train'):
    images = images_in_dir(os.path.join(image_dir, subset))
    if len(images) < 1:
        print('No images found in {} '.format(image_dir))
        return
    invalid = []
    for im in images:
        try:
            skimage.io.imread(im)
            print('{} is valid.'.format(im))
        except Exception:
            invalid.append(im)

    print('Invalid images: {}'.format(invalid))


def display_masks(image_dir, subset='train', images_num=10):
    print('Display image masks')
    dataset = PersonDataset()
    dataset.load_person(image_dir, subset)
    print('Prepare...')
    dataset.prepare()
    print('Randomly select {} images to display'.format(images_num))
    image_ids = np.random.choice(dataset.image_ids, images_num)
    print(len(dataset.image_ids))
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    file_name = ''
    if image_path:
        print("Running on {}".format(args.image))
        image = skimage.io.imread(args.image)
        r = model.detect([image], verbose=1)[0]
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            success, image = vcapture.read()
            if success:
                image = image[..., ::-1]
                r = model.detect([image], verbose=0)[0]
                splash = color_splash(image, r['masks'])
                splash = splash[..., ::-1]
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect person.')
    parser.add_argument("command", metavar="<command>", help="'train' or 'splash' or 'validate' or 'display' ")
    parser.add_argument('--dataset', required=False, metavar="/path/to/person/dataset/",
                        help='Directory of the Person dataset')
    parser.add_argument('--weights', required=False, metavar="/path/to/weights.h5", default='coco',
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False, default=DEFAULT_LOGS_DIR, metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False, metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False, metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    if args.command not in ['train', 'splash', 'validate', 'display']:
        print("'{}' is not recognized. " "Use 'train' or 'splash', 'validate', 'display' ".format(args.command))
        exit(0)

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, "Provide --image or --video to apply color splash"
    elif args.command == 'validate':
        assert args.dataset, "Argument --dataset is required for validation"
        validate_images(args.dataset)
        exit(0)
    elif args.command == 'display':
        assert args.dataset, "Argument --dataset is required for display"
        display_masks(args.dataset)
        exit(0)

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PersonConfig()
    else:
        class InferenceConfig(PersonConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)
    model.keras_model.summary()
    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, config)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image, video_path=args.video)
