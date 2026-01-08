import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '5'  # YOLO会默认占用一部分第一张能看到的卡的显存

from ultralytics import YOLO


model = YOLO('weights/best.pt')  # load a pretrained model (recommended for training)
# model = YOLO('')
# model = YOLO('')


model.val(data="gen1.yaml",device="cuda:0")
