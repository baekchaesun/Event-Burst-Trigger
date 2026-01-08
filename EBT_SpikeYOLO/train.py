import os
from ultralytics import YOLO

os.environ['WANDB_DISABLED'] = 'true'
fr_dict = {}

WEIGHTS_PATH = ""

if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
    model = YOLO(WEIGHTS_PATH)
else:
    model = YOLO("snn_yolov8s.yaml")

model.train(data="gen1.yaml",device="cuda:0")