from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import os.path as osp
import cv2
import logging
import argparse
import motmetrics as mm
import numpy as np
import torch
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets

from collections import defaultdict
import pandas as pd
import pickle

from tracking_utils.utils import mkdir_if_missing
from opts import opts


def write_results(filename, results, data_type,online_im):

    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)

    logger.info('save results to {}'.format(filename))
def mouse_draw(img0):
    pts = []  # for storing points
    # :mouse callback function
    def draw_roi(event, x, y, flags, param):
        img2 = img.copy()

        if event == cv2.EVENT_LBUTTONDOWN:  # Left click, select point
            pts.append((x, y))

        if event == cv2.EVENT_RBUTTONDOWN:  # Right click to cancel the last selected point
            pts.pop()

        if len(pts) > 0:
        # Draw the last point in pts
            cv2.circle(img2, pts[-1], 3, (0, 0, 255), -1)

        if len(pts) > 1:
        #
            for i in range(len(pts) - 1):
                cv2.circle(img2, pts[i], 5, (0, 0, 255), -1)  # x ,y is the coordinates of the mouse click place
                cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1], color=(255, 0, 0), thickness=2)

        cv2.imshow('image', img2)

    # Create images and windows and bind windows to callback functions
    img = img0
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)

    print("[INFO] Click the left button: select the point, right click: delete the last selected point, click the middle button: determine the ROI area")
    print("[INFO] Press ‘S’ to determine the selection area and save it")
    print("[INFO] Press ESC to quit")
    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == 27:
    #         break
    #     if key == ord("s"):
    #         saved_data = {
    #             "ROI": pts
    #         }
    #         joblib.dump(value=saved_data, filename="config.pkl")
    #         print("[INFO] ROI coordinates have been saved to local.")
    #     break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pts
def write_box_loitering(img0,pts):
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img0, [pts], True, (255, 255, 0))

def warning_loitering(img0,pts):
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img0, [pts], True, (0, 0, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_text = 100
    x_text = int((img0.shape[1]) / 2) - 300
    cv2.putText(img0, 'Warning', (x_text, y_text), font, 4, (0, 255, 255), 2, cv2.LINE_AA)


def write_bbox(img0,status,pts):
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img0, [pts], True, (0, 0, 255))
    if(status == 'warning'):
       # polygon = Polygon([(290,400), (1140,1000), (1630,580), (850,310)])
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_text = int((img0.shape[0])/2)
        x_text = int((img0.shape[1])/2)-300
        cv2.putText(img0, 'Warning', (x_text,y_text), font, 4, (0, 0,255), 2, cv2.LINE_AA)
    elif(status == 'normal'):
        cv2.polylines(img0, [pts], True, (0, 255, 0))

def extract_person(tlwh, tid, img0, frame_id):
    #path = "C:/Users/Windows 10/PycharmProjects/test/FairMOT-master/results/frame"
    path_peron = "/home/tuannguyen/Desktop/FairMOT-master/results/person"
    if (os.path.exists(path_peron) == False):
        os.mkdir(path_peron)
    path1 = path_peron + '/' + str(tid)
    if (os.path.exists(path1) == False):
        os.mkdir(path1)
    x, y, w, h = tlwh
    x,y,w,h = int(x), int(y), int(w),int(h)
    if y + h > img0.shape[0]:
        y_max = img0.shape[0]
    else:
        y_max = y + h
    if x + w > img0.shape[1]:
        x_max = img0.shape[1]
    else:
        x_max = x + w
    img_save = img0[y:y_max, x:x_max]
    cv2.imwrite(path1 + '/' + str(tid) + '-' + str(frame_id) + '.jpg', img_save)

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    pts = []
    pts2 = []
    count = 0
    count_loitering = 0
    key_id = []
    key_count = []
    dict_True = {}
    dict_False = {}
    dict_feat = defaultdict(list)
    dict_feat_final = defaultdict(list)
    list_feat = []
    for path, img, img0 in dataloader:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        if frame_id == 0:
            # img0 = cv2.resize(img0,(int(img0.shape[1]/2),int(img0.shape[0]/2)))
            pts = mouse_draw(img0)
            pts2 = mouse_draw(img0)
        ###
        write_bbox(img0, 'normal', pts)
        write_box_loitering(img0, pts2)
        # run tracking

        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets, id_feat = tracker.update(blob, img0)

        online_tlwhs = []
        online_ids = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

            if tid not in dict_True:
                dict_True[tid] = 0
            if tid not in dict_False:
                dict_False[tid] = 0
            ###
            x, y, w, h = tlwh
            point2 = Point((x + int(w / 2)), y + h)
            polygon = Polygon(pts)
            polygon2 = Polygon(pts2)
            if (polygon.contains(point2) == True):
                write_bbox(img0, 'warning', pts)
            if (polygon2.contains(point2) == True):
                dict_True[tid] += 1
                if (dict_True[tid] > 100):
                    warning_loitering(img0, pts2)
                    extract_person(tlwh, tid, img0, frame_id)

            if (polygon2.contains(point2) == False):
                dict_False[tid] += 1
                if (dict_False[tid] > 50):
                    dict_True[tid] = 0

            for i in range(len(online_ids)):
                print(id_feat.shape[0], len(online_ids))
                key = online_ids[i]
                # json_type = pd.Series(id_feat[i]).to_json(orient='values')
                dict_feat[key].append(id_feat[i])

            print(online_ids, id_feat.shape)
            print("-------------------------------------------------------------------------------")
            # for feat in id_feat:
            #     print(feat)
            ###
            # extract_person(tlwh,tid,img0,count)
            # count = count + 1
            ###

        timer.toc()
        ###

        ###

        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    # print(dict_feat)
    # print(dict_feat_final)
    for key, value in dict_feat.items():
        length = len(dict_feat[key])
        dict_feat_final[key].append(value[0])
        dict_feat_final[key].append(value[int(length / 2)])
        dict_feat_final[key].append(value[-1])

    with open("sample.pkl", "wb") as outfile:
        pickle.dump(dict_feat_final, outfile)
    write_results(result_filename, results, data_type, online_im)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    for seq in seqs:
        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(osp.join(data_root, seq, 'img1'), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        if save_videos:
            output_video_path = osp.join(output_dir, '{}.mp4'.format(seq))
            cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}'.format(output_dir, output_video_path)
            os.system(cmd_str)
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, 1.0 / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        data_root = os.path.join(opt.data_dir, 'MOT16/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name='MOT15_val_all_dla34',
         show_image=False,
         save_images=False,
         save_videos=False)
