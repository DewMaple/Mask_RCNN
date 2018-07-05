import json
import os
import shutil

import numpy as np
import skimage.draw
import skimage.io

import utils


class PersonDataset(utils.Dataset):
    def load_person(self, dataset_dir, subset, source='person'):
        self.add_class(source, 1, "person")
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        annotations = json.load(open(os.path.join(dataset_dir, "regions.json")))

        annotations = [a for a in annotations if a['regions']]
        print('annotations length is {}'.format(len(annotations)))
        for i, a in enumerate(annotations):
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            image_path = os.path.join(dataset_dir, a['filename'])

            height, width = a['height'], a['width']

            self.add_image(source, image_id=a['filename'], path=image_path, width=width, height=height,
                           polygons=polygons)

        print('{} annotations of images loaded'.format(len(annotations)))

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        if image_info["source"] != "person":
            return super(self.__class__, self).load_mask(image_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "person":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train_data_2_val(dataset_dir, tar_dir):
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    train_data = json.load(open(os.path.join(train_dir, "regions.json")))
    print('origin train dataset size: ', len(train_data))

    val_data = json.load(open(os.path.join(val_dir, "regions.json")))

    choose_train = np.random.choice(train_data, 10000).tolist()
    print('move {} train data to val data'.format(len(choose_train)))

    tar_train = os.path.join(tar_dir, 'train')
    tar_val = os.path.join(tar_dir, 'val')

    print('copy chosen train data to tar val')
    for a in choose_train:
        shutil.copy(os.path.join(train_dir, a['filename']), tar_val)

    print('remove chosen data from train data')
    train_data = [d for d in train_data if d not in choose_train]

    print('copy train data to tar train')
    for a in train_data:
        shutil.copy(os.path.join(train_dir, a['filename']), tar_train)

    print('save new train data regions.json')
    with open(os.path.join(tar_train, 'regions.json'), 'w') as f:
        json.dump(train_data, f)

    print('copy val data to tar val')
    for a in val_data:
        shutil.copy(os.path.join(val_dir, a['filename']), tar_val)

    print('add val data from chosen')
    val_data = val_data + choose_train

    print('save new val data regions.json')
    with open(os.path.join(tar_val, 'regions.json'), 'w') as f:
        json.dump(val_data, f)

    print('train data length is {}, val data length is {}'.format(len(train_data), len(val_data)))


if __name__ == '__main__':
    train_data_2_val('samples/person/coco', 'dataset/person')
