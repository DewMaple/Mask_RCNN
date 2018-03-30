import os

import pylab
from pycocotools.coco import COCO

pylab.rcParams['figure.figsize'] = (8.0, 10.0)


def download_person_dataset(dataset_dir=".", year=2017):
    download(dataset_dir, 'val', year, ['person'])


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
        except Exception:
            count += 1
            print("Retry {} times".format(count))

    print("{} data downloaded, {} images. ".format(subset, len(img_ids)))


if __name__ == '__main__':
    download_person_dataset('/Users/administrator/workspace/Mask_RCNN/samples/person/coco', 2017)
