#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import pickle
import numpy as np
import json
from tqdm import tqdm

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import pycocotools.mask as mask_util

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--video-id',
        dest='video_id',
        default=0,
        type=int
    )
    parser.add_argument(
        '--num-split',
        dest='num_split',
        default=0,
        type=int
    )
    parser.add_argument(
        '--split-id',
        dest='split_id',
        default=0,
        type=int
    )
    parser.add_argument(
        '--task',
	dest='task',
	default='mask',
	type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)

    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    args.im_or_folder = 'input/{:03}-frames'.format(args.video_id)

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    im_list = list(im_list)
    im_list.sort()
    im_list = [im_list[i] for i in range(args.split_id, len(im_list), args.num_split)] 

    print('Process {} on video {} {} for {} frames'.format(args.task, args.video_id, args.split_id, len(im_list)))
    
    result_path = '{}/detectron-{}-{}-{}.pkl'.format(args.output_dir, args.task, args.video_id, args.split_id)
    if os.path.exists(result_path):
	result = pickle.load(open(result_path, 'rb'))
    else:    
        result = {}
    for idx, im_name in tqdm(enumerate(im_list)):
        fid = int(im_name.split('/')[-1][:-4])
        if (args.video_id, fid) in result:
	    continue        

        logger.info('Split {} processing {}th image: {} '.format(args.split_id, idx, im_name))
        im = cv2.imread(im_name)
        if im is None:
            continue
        # Crop image 
        # crop_bbox = crop_bbox_dict[os.path.basename(im_name)]
        # im_crop = im[crop_bbox[1] : crop_bbox[3], crop_bbox[0] : crop_bbox[2]]

        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )

        # Update result with crop_bbox
        # cls_boxes_update = []
        # cls_segms_update = []
        # for box_list in cls_boxes:
        #     box_list_update = []
        #     for box in box_list:
        #         box[0] += crop_bbox[0]
        #         box[2] += crop_bbox[0]
        #         box[1] += crop_bbox[1]
        #         box[3] += crop_bbox[1]
        #         box_list_update.append(box)
        #     cls_boxes_update.append(box_list_update)
        # for segm_list in cls_segms:
        #     segm_list_update = []
        #     for mask in segm_list:
        #         mask_full = np.zeros((1080, 1920), dtype=np.uint8)
        #         mask_full[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]] = mask_util.decode(mask)
        #         segm_list_update.append(mask_util.encode(np.asfortranarray(mask_full)))
        #     cls_segms_update.append(segm_list_update)
        # cls_boxes = cls_boxes_update
        # cls_segms = cls_segms_update

        #logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        #for k, v in timers.items():
        #    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
       
        result[(args.video_id, fid)] = {'bbox': cls_boxes, 'segm': cls_segms, 'keyp': cls_keyps}

        # vis_utils.vis_one_image(
        #     im[:, :, ::-1],  # BGR -> RGB for visualization
        #     im_name,
        #     args.output_dir,
        #     cls_boxes,
        #     cls_segms,
        #     cls_keyps,
        #     dataset=dummy_coco_dataset,
        #     box_alpha=0.3,
        #     show_class=True,
        #     thresh=args.thresh,
        #     kp_thresh=args.kp_thresh,
        #     ext=args.output_ext,
        #     out_when_no_box=args.out_when_no_box
        # )
        if idx % 100 == 0:
            pickle.dump(result, open(result_path, 'wb'), protocol=2)

    pickle.dump(result, open(result_path, 'wb'), protocol=2)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
