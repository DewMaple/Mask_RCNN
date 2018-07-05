import json
import os
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool

import pylab
from pycocotools.coco import COCO

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


def download_person_dataset(dataset_dir=".", year=2017):
    download(dataset_dir, 'train', year, ['person'])
    # download(dataset_dir, 'train', year, ['person'])


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
    if threads_num == 1:
        download_from_coco(coco, image_ids, tar_dir)
    else:
        img_len = len(image_ids)
        batch_size = img_len // threads_num
        batch_size = batch_size if img_len % threads_num == 0 else batch_size + 1

        pool = Pool(threads_num)
        args = []
        for i in range(threads_num):
            start = i * batch_size
            end = min(start + batch_size, img_len)
            args.append(image_ids[start:end])

        pool.starmap(download_from_coco, zip(repeat(coco), args, repeat(tar_dir)))


def download(data_dir, subset, year, cat_nms):
    data_type = '{}{}'.format(subset, year)
    ann_file = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
    coco = COCO(ann_file)

    cat_ids = coco.getCatIds(catNms=cat_nms)
    img_ids = coco.getImgIds(catIds=cat_ids)

    subset_dir = os.path.join(data_dir, subset)
    os.makedirs(subset_dir, exist_ok=True)

    print('save dataset to {}.'.format(subset_dir))

    # copy_files(coco, img_ids[:100], subset_dir, 'dataset/person/' + subset)
    # create_annotation(coco, img_ids[:100], 'dataset/person/' + subset, cat_ids=cat_ids)
    # download_from_coco(coco, image_ids=img_ids, tar_dir=subset_dir)
    multi_thread_downloader(coco, image_ids=img_ids, tar_dir=subset_dir, threads_num=4)
    create_annotation(coco, image_ids=img_ids, tar_dir=subset_dir, cat_ids=cat_ids)


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

    img_ids = ann['image_id']
    img = coco.loadImgs(img_ids)[0]
    return {
        'filename': img['file_name'],
        'regions': regions,
        'size': img['height'] * img['width'],
        'height': img['height'],
        'width': img['width']
    }


def _merge(a1, a2):
    l1 = len(a1['regions'])
    l2 = len(a2['regions'])
    for i in range(l2):
        a1['regions'][i + l1] = a2['regions'][i]

    return a1


def _merge_by_filename(anns):
    if len(anns) < 2:
        return anns
    annotations = sorted(anns, key=lambda c: c['filename'])
    ann_list = []
    pre = annotations[0]
    for a in annotations[1:]:
        if pre['filename'] == a['filename']:
            pre = _merge(pre, a)
        else:
            ann_list.append(pre)
            pre = a
    return ann_list


def create_annotation(coco, image_ids, tar_dir, cat_ids):
    ann_ids = coco.getAnnIds(imgIds=image_ids, catIds=cat_ids)
    print('img_ids len is {}'.format(len(image_ids)))
    print('ann len is {}'.format(len(ann_ids)))
    anns = coco.loadAnns(ann_ids)
    annotations = []
    for ann in anns:
        annotations.append(_shape_coco_2_vgg(ann, coco))

    annotations = _merge_by_filename(annotations)
    with open(os.path.join(tar_dir, 'regions.json'), 'w') as f:
        json.dump(annotations, f)
    print("Annotation length is {}".format(len(annotations)))


if __name__ == '__main__':
    download_person_dataset('/Users/administrator/workspace/Mask_RCNN/dataset/person', 2017)
