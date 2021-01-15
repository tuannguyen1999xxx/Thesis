import cv2
from lib.tracking_utils.timer import Timer

tracking_model_name = 'fair_mot'
url = '0.0.0.0:8001'
protocol = 'grpc'
batch_size = 1
is_async = True
is_streaming = True


cap = cv2.VideoCapture('./data/MOT16-03.mp4')
frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
count = 0
w, h = 1920, 1080
input_size=(576, 320)

timer = Timer()
