
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
from tracker.multitracker import JDETracker

import torch

logger.setLevel(logging.INFO)


def demo(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)
    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)

    device = torch.device("cuda")
    blob = torch.randn(3, 3, 608, 1088).to(device)

    output_onnx = 'model.onnx'
    input_names = ["input"]
    output_names = ['3852', '3855', '3858', '3861']
    dynamic_axes ={'input': {0: 'batch_size'}, '3852': {0: 'batch_size'}, '3855': {0: 'batch_size'}, '3858': {0: 'batch_size'}, '3861': {0: 'batch_size'}}
    frame_rate = dataloader.frame_rate


    tracker = JDETracker(opt, frame_rate=frame_rate)
    model = tracker.model

    torch_out = torch.onnx.export(model, blob, output_onnx, export_params=True, verbose=False,

                              input_names=input_names, output_names=output_names, opset_version=10, dynamic_axes=dynamic_axes)

if __name__ == '__main__':

    opt = opts().init()

    demo(opt)


