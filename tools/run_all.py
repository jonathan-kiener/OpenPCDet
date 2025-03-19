import argparse
import glob
from pathlib import Path

import numpy as np
import torch

import pcdet
#from pcdet.datasets import __all__ as datasets
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--save_path', type=str, default=None, 
                        help='specify the output folder for the bounding boxes')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Running Checkpoint on All Data-------------------------')
    dataset = datasets[cfg.DATA_CONFIG.DATASET](dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger)
    #dataset = KittiDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, logger=logger)

    logger.info(f'Total number of samples: \t{len(dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            print(f"RESULT: {len(pred_dicts)} Predictions:")

            #for i, pred in enumerate(pred_dicts):
                #np.save(args.save_path + f"{}", pred_dicts[0]['pred_boxes'].cpu().numpy())

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
