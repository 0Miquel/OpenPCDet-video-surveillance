import argparse
import copy
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_yaml_file


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='specify the config for demo')
    # "cfgs/custom_models/centerpoint.yaml"
    # "cfgs/nuscenes_models/final_models/voxel_centerpoint_human_orientationless.yaml"
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    # '../checkpoints/custom_models/custom_5sweep.pth'
    # '../checkpoints/orientationless.pth'
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=args.workers, logger=logger, training=False
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            if idx % 5 == 0:
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
                V.draw_scenes(
                    points=data_dict['points'][:, 1:],
                    ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'],
                    ref_labels=pred_dicts[0]['pred_labels'],
                    gt_boxes=data_dict['gt_boxes'][0]
                )

                if not OPEN3D_FLAG:
                    mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

