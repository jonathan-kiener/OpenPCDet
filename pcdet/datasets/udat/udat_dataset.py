import pickle
import numpy as np
from skimage import io

from ..kitti.kitti_dataset import KittiDataset

class UdatDataset(KittiDataset):

    def get_image_raw(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists()
        return img_file

    def get_image(self, idx):  
        image = io.imread(self.get_image_raw(idx))
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        image = self.get_image_raw(idx)
        return np.array(io.imread(image).shape[:2], dtype=np.int32)


def create_kitti_infos(dataset_cfg, class_names, save_path, workers=4):
    dataset = UdatDataset(dataset_cfg=dataset_cfg, class_names=class_names, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from easydict import EasyDict
        from pathlib import Path

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        dataset_path = Path(dataset_cfg.DATA_PATH)

        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Object'],
            save_path=dataset_path
        )
