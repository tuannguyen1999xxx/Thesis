import os
import os.path as osp
import shutil
import json
import time
import argparse
from argparse import Namespace

import pandas as pd
import cv2
import torch
import numpy as np
from tensorrtserver.api import *

import lib.datasets.dataset.jde as datasets
from lib.tracking_utils import visualization as vis
from lib.tracker.multitracker import JDETracker
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer

from utils.utils import Logger
# from heigh_estimation import CameraCalibration

opt = None

def mkdirifmissing(path):
    if osp.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


class FullPipeline():
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.tracker = JDETracker(opt, frame_rate=30)
        self.log = Logger()
        self.meta = {}
        # self.camera = CameraCalibration(self.opt)
        # self.camera.load_params(self.opt.cam_params)

    def init_meta(self, in_vid):
        self.cap = cv2.VideoCapture(in_vid)
        self.meta['video_name'] = osp.basename(in_vid)
        self.meta['frame_rate'] = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.meta['vw'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.meta['vh'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.meta['vn'] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.meta['cap_id'] = 0
        self.meta['p_count'] = {}
    
    def _append_tracked(self, results, tid, frame_id, x, y, w, h, height):
        if tid not in results:
            results[tid] = {
                'frame_id': [],
                'x': [],
                'y': [],
                'w': [],
                'h': [],
                'height': []
            }
        results[tid]['frame_id'].append(tid)
        results[tid]['x'].append(x)
        results[tid]['y'].append(y)
        results[tid]['w'].append(w)
        results[tid]['h'].append(h)
        results[tid]['height'].append(height)
        
    def crop_human(self, output='crop', line=None):        
        path = osp.join(output, self.meta['video_name'])

        line = line.strip()
        try:
            frame_id, p_id, img_w, img_h, x, y, w, h = line.split('/')
            frame_id = int(frame_id)
            p_id = int(p_id)
            img_w = int(img_w)
            img_h = int(img_h)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            wratio = self.meta['vw'] / img_w
            hratio = self.meta['vh'] / img_h

            actual_loc = [int(x*wratio), int(y*hratio), int((x+w)*wratio), int((y+h)*hratio)]
            res = False
            while self.meta['cap_id'] < frame_id:
                res, img0 = self.cap.read()
                self.meta['cap_id'] += 1
                if not res:
                    return

            if res:
                human_img = img0[actual_loc[1]:actual_loc[3], actual_loc[0]:actual_loc[2], :]
                if human_img.shape[0] == 0 or human_img.shape[1] == 0 or human_img.shape[2] == 0:
                    return
                if p_id not in self.meta['p_count']:
                    self.meta['p_count'][p_id] = 0
                    mkdirifmissing(osp.join(path, str(p_id)))
                self.meta['p_count'][p_id] += 1
                cv2.imwrite(osp.join(path, str(p_id), str(self.meta['p_count'][p_id]) + '.png'), human_img)
        except BaseException as e:
            print(f'Crop human {e}')
    
    def _in_ROI(self, frame_width, frame_height, x1, y1, w, h):
        return y1 > frame_height * 0.05 and (y1 + h) < frame_height * 0.9 \
                        and x1 > frame_width * 0.05 and (x1 + w) < frame_width * 0.9
    def letterbox(self, img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh

    def process(self, img0, input_size, w, h, frame_id, timer, show_image=False):
        img0 = cv2.resize(img0, (w, h))

        # Padded resize
        
        width = input_size[0]
        height = input_size[1]
        img, _, _, _ = self.letterbox(img0, height=height, width=width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        
        if frame_id % 20 == 1:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, \
                        1. / max(1e-5, timer.average_time)))
        if frame_id % self.opt.batch_size == 0:
            input_batch = []
        input_batch.append(img)

        # run tracking
        print("frame_id % self.opt.batch_size: ", frame_id, self.opt.batch_size, frame_id % self.opt.batch_size)
        timer.tic()
        if frame_id % self.opt.batch_size == self.opt.batch_size - 1:
            online_targets, remove_tracked = self.tracker.update(input_batch, img0)
            print(remove_tracked)
            online_tlwhs = []
            online_ids = []
                    
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                tid_height = 0
                x1, y1, w, h = tlwh

                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            timer.toc()
            if show_image:
                online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                        fps=1. / max(1e-5, timer.average_time))
                cv2.imshow('online_im', online_im)
                cv2.waitKey(1)
        frame_id += 1
