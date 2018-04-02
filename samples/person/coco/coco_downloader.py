import json
import os
import shutil
import threading
import time

import pylab
from pycocotools.coco import COCO

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


def download_person_dataset(dataset_dir=".", year=2017):
    # download(dataset_dir, 'train', year, ['person'])
    download(dataset_dir, 'val', year, ['person'])


def copy_files(coco, image_ids, src_dir, tar_dir):
    images = coco.loadImgs(image_ids)
    for im in images:
        shutil.copy(os.path.join(src_dir, im['file_name']), tar_dir)


def download_from_coco(coco, image_ids, tar_dir):
    count = 0
    while True:
        try:
            coco.download(tarDir=tar_dir, imgIds=image_ids)
            break
        except Exception as e:
            count += 1
            print(e)
            print("Retry {} times".format(count))
        time.sleep(60)
    print("{} data downloaded, {} images. ".format(tar_dir, len(image_ids)))


def multi_thread_downloader(coco, image_ids, tar_dir, threads_num=1):
    class CocoDownloader(threading.Thread):
        def __init__(self, coco, image_ids, tar_dir, thread_id):
            threading.Thread.__init__(self)
            self.coco = coco
            self.image_ids = image_ids
            self.tar_dir = tar_dir
            self.thread_id = thread_id

        def run(self):
            print("Thread {} is running".format(self.thread_id))


def download(data_dir, subset, year, cat_nms):
    data_type = '{}{}'.format(subset, year)
    ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
    coco = COCO(ann_file)

    cat_ids = coco.getCatIds(catNms=cat_nms)
    img_ids = coco.getImgIds(catIds=cat_ids)

    subset_dir = os.path.join(data_dir, subset)
    os.makedirs(subset_dir, exist_ok=True)

    print('save dataset to {}.'.format(subset_dir))

    copy_files(coco, img_ids[:100], subset_dir, 'dataset/person/' + subset)
    create_annotation(coco, img_ids[:100], 'dataset/person/' + subset, cat_ids=cat_ids)
    # download_from_coco(coco, image_ids=img_ids, tar_dir=subset_dir)
    # create_annotation(coco, image_ids=img_ids, tar_dir=subset_dir, cat_ids=cat_ids)


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
    print(ann)

    regions = {}
    import numpy as np
    for i, seg in enumerate(segs):
        if type(seg) is not list:
            continue
        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        polygon = pylab.Polygon(poly)
        xy = polygon.get_xy()

        regions[i] = {
            'region_attributes': {},
            'shape_attributes': {
                'all_points_x': xy[:, 0].tolist(),
                'all_points_y': xy[:, 1].tolist(),
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


def create_annotation(coco, image_ids, tar_dir, cat_ids):
    ann_ids = coco.getAnnIds(imgIds=image_ids, catIds=cat_ids)
    print('img_ids len is {}'.format(len(image_ids)))
    print('ann len is {}'.format(len(ann_ids)))
    anns = coco.loadAnns(ann_ids)
    annotations = []
    for ann in anns:
        annotations.append(_shape_coco_2_vgg(ann, coco))

    with open(os.path.join(tar_dir, 'regions.json'), 'w') as f:
        json.dump(annotations, f)


if __name__ == '__main__':
    download_person_dataset('/Users/administrator/workspace/Mask_RCNN/samples/person/coco', 2017)
