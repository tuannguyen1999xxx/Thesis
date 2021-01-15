import cv2
from tensorrtserver.api import *
from lib.opts import opts
from utils.utils import Logger
from utils import cfg
from utils.human_tracking import FullPipeline


if __name__ == '__main__':
    opt = opts().init()
    print('___init_model___')
    pipeline = FullPipeline(opt)
    print('___start_process___')


    frame_id = 0
    # self.log.init_logger()
    while True:
        res, img0 = cfg.cap.read()  # BGR
        
        pipeline.process(img0, cfg.input_size, cfg.w, cfg.h, frame_id, cfg.timer, show_image=True)
    # self.log.close()
    # pipeline.crop_human(args.input_video, 'results.txt')
