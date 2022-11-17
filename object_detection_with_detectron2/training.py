# Usage : python3.8 training.py --outdir out_test/ --arch detr  \
#         --config configs/detr_256_6_6_torchvision.yaml  \
#         --weight pretrained_model/converted_model.pth --epochs 1
#       use dyhead_swint_atss_fpn_2x_ms.yaml for dyhead-swint

import detectron2

from detectron2.utils.logger import setup_logger
setup_logger()
import sys
sys.path.append('detr')
sys.path.append('DynamicHead/')
# import some common libraries
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# from detectron2.engine import DefaultTrainer
from d2.train_net import Trainer
from d2.detr import add_detr_config

from utils.dyhead_trainer import DyHeadTrainer
import matplotlib.pyplot as plt

import argparse

from dyhead import add_dyhead_config
from extra import add_extra_config

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--outdir", required=True,	help="output path  for the model")
ap.add_argument('-a', '--arch', default='detr', choices=['detr', 'dyhead-fpn', 'dyhead-swint'], help='Choose different transformer based object detection architecture')
ap.add_argument("-c", "--config", required=True,	help="config file for the model")
ap.add_argument("-w", "--weight", required=True,	help="model weight path")
ap.add_argument("-e", "--epochs", type=int, help="No of Epochs for training")
args = vars(ap.parse_args())

register_coco_instances("custom_train",
                        {},
                        "../custom_balloon/annotations/custom_train.json",
                        "../custom_balloon/train2017/")

register_coco_instances("custom_val",
                        {},
                        "../custom_balloon/annotations/custom_val.json",
                        "../custom_balloon/val2017/")


LABELS = ["balloon"]

for keyword in ['train', 'val']:
  MetadataCatalog.get('custom_{}'.format(keyword)).set(thing_classes=LABELS)

custom_metadata = MetadataCatalog.get("custom_train")

dataset_dicts = DatasetCatalog.get("custom_train")
for d in random.sample(dataset_dicts, 1):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=custom_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(img[:, :, ::-1])  # BGR to RGB
    ax[0].set_title('Original Image ')
    ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
    ax[1].set_title('Detected Image')
    plt.show()


outdir_path = args['outdir']
weight = args['weight']
config = args['config']
epochs = args['epochs']
print("outdir path : {}".format(outdir_path))
print("weight file : {}".format(weight))
print("config file : {}".format(config))

cfg = get_cfg()
if args['arch'] == 'detr':
    add_detr_config(cfg)
elif args['arch'] == 'dyhead-fpn' or args['arch'] == 'dyhead-swint':
    add_dyhead_config(cfg)
    add_extra_config(cfg)

cfg.merge_from_file(config)

cfg.DATASETS.TRAIN = ("custom_train",)
cfg.DATASETS.TEST = ("custom_val",)

cfg.OUTPUT_DIR = outdir_path
#print("cfg: {}".format(cfg))

if args["arch"] == 'detr':
    cfg.MODEL.WEIGHTS = weight
    cfg.MODEL.DETR.NUM_CLASSES = len(LABELS)
elif args['arch'] == 'dyhead-fpn':
    cfg.MODEL.WEIGHTS = weight #'pretrained_model/dyhead_r50_atss_fpn_1x.pth'
    cfg.MODEL.ATSS.NUM_CLASSES = len(LABELS)
elif args['arch'] == 'dyhead-swint':
    cfg.MODEL.WEIGHTS = weight #'pretrained_model/dyhead_swint_atss_fpn_2x_ms.pth'
    cfg.MODEL.ATSS.NUM_CLASSES = len(LABELS)

cfg.DATALOADER.NUM_WORKERS = 2
if args["arch"] == 'dyhead-swint':
    cfg.SOLVER.IMS_PER_BATCH = 1
else:
    cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = epochs
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(LABELS)  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = None
if args['arch'] == 'detr':
    trainer = Trainer(cfg)
elif args['arch'] == 'dyhead-fpn' or args['arch'] == 'dyhead-swint':
    trainer = DyHeadTrainer(cfg)

trainer.resume_or_load(resume=False)
trainer.train()
