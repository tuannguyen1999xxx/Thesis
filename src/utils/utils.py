import tensorrtserver.api.model_config_pb2 as model_config
import numpy as np
from tensorrtserver.api import *
from PIL import Image
import cv2
import time
import ctypes

class Logger():
    def __init__(self):
        super().__init__()
        
    def init_logger(self):
        self.writer = open('results.txt', 'w')
    
    def log(self, msg):
        self.writer.write(msg + '\n')
    
    def close(self):
        self.writer.close()


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper

