import json
import os
import time

import pylab
from pycocotools.coco import COCO

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


def download_person_dataset(dataset_dir=".", year=2017):
    download(dataset_dir, 'train', year, ['person'])


def download(data_dir, subset, year, cat_nms):
    data_type = '{}{}'.format(subset, year)
    ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
    coco = COCO(ann_file)

    cat_ids = coco.getCatIds(catNms=cat_nms)
    img_ids = coco.getImgIds(catIds=cat_ids)

    subset_dir = os.path.join(data_dir, subset)
    os.makedirs(subset_dir, exist_ok=True)
    print('save dataset to {}.'.format(subset_dir))
    count = 0

    while True:
        try:
            coco.download(tarDir=subset_dir, imgIds=img_ids)
            break
        except Exception as e:
            count += 1
            print(e)
            print("Retry {} times".format(count))
        time.sleep(60)
    print("{} data downloaded, {} images. ".format(subset, len(img_ids)))

    create_annotation(subset_dir, coco, img_ids)


def _shape_coco_2_vgg(ann, coco):
    # VGG Image Annotator saves each image in the form:
    # { 'filename': '28503151_5b5b7ec140_b.jpg',
    #   'regions': {
    #       '0': {
    #           'region_attributes': {},
    #           'shape_attributes': {
    #               'all_points_x': [...],
    #               'all_points_y': [...],
    #               'name': 'polygon'}},
    #       ... more regions ...
    #   },
    #   'size': 100202
    # }
    segs = ann['segmentation']

    regions = {}
    for i, seg in enumerate(segs):
        idx = int(len(seg) / 2)
        regions[i] = {
            'region_attributes': {},
            'shape_attributes': {
                'all_points_x': seg[0:idx],
                'all_points_y': seg[idx:],
                'name': 'polygon'
            }
        }

    img_id = ann['image_id']
    img = coco.loadImgs(img_id)[0]
    return {
        'filename': img['file_name'],
        'regions': regions,
        'size': img['height'] * img['width']
    }


def create_annotation(tar_dir, coco, img_ids):
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    print('img_ids len is {}'.format(len(img_ids)))
    print('ann len is {}'.format(len(ann_ids)))
    anns = coco.loadAnns(ann_ids)
    annotations = []
    for ann in anns:
        annotations.append(_shape_coco_2_vgg(ann, coco))

    with open(os.path.join(tar_dir, 'regions.json'), 'w') as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    download_person_dataset('/Users/administrator/workspace/Mask_RCNN/samples/person/coco', 2017)
