# Usage: python3.8 inference.py --outdir out_test --arch detr \
#        --config configs/detr_256_6_6_torchvision.yaml --save True
#

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import sys
sys.path.append('detr')
# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
# from detectron2.engine import DefaultTrainer
from d2.train_net import Trainer
from d2.detr import add_detr_config

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from dyhead import add_dyhead_config
from extra import add_extra_config

import matplotlib.pyplot as plt
import time
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--outdir", required=True,	help="output path  for the model")
ap.add_argument('-a', '--arch', default='detr', choices=['detr', 'dyhead-fpn', 'dyhead-swint'], help='Choose different transformer based object detection architecture')
ap.add_argument("-c", "--config", required=True,	help="config file for the model")
ap.add_argument("-s", '--save', default=False, type=bool, help='save predicted output')
args = vars(ap.parse_args())

outdir_path = args['outdir']
config = args['config']
print("outdir path : {}".format(outdir_path))
print("config file : {}".format(config))

cfg = get_cfg()
if args['arch'] == 'detr':
    add_detr_config(cfg)
elif args['arch'] == 'dyhead-fpn' or args['arch'] == 'dyhead-swint':
    add_dyhead_config(cfg)
    add_extra_config(cfg)

cfg.merge_from_file(config)

LABELS = ["balloon"]

cfg.DATASETS.TRAIN = ("custom_train",)
cfg.DATASETS.TEST = ("custom_val",)

cfg.OUTPUT_DIR = outdir_path

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
if args["arch"] == 'detr':
    cfg.MODEL.DETR.NUM_CLASSES = len(LABELS)
elif args['arch'] == 'dyhead-fpn' or args['arch'] == 'dyhead-swint':
    cfg.MODEL.ATSS.NUM_CLASSES = len(LABELS)

predictor = DefaultPredictor(cfg)

register_coco_instances("custom_train",
                        {},
                        "../custom_balloon/annotations/custom_train.json",
                        "../custom_balloon/train2017/")

register_coco_instances("custom_val",
                        {},
                        "../custom_balloon/annotations/custom_val.json",
                        "../custom_balloon/val2017/")



for keyword in ['train', 'val']:
  MetadataCatalog.get('custom_{}'.format(keyword)).set(thing_classes=LABELS)

custom_metadata = MetadataCatalog.get("custom_val")

dataset_dicts = DatasetCatalog.get("custom_val")

# Modification
#dataset_name = cfg.DATASETS.TRAIN[0]
#custom_metadata = MetadataCatalog.get(dataset_name)

threshold = 0.7

#dataset_dicts = DatasetCatalog.get("custom_val")

def filter_predictions_from_outputs(outputs,
                                    threshold=0.7,
                                    verbose=True):
  predictions = outputs["instances"].to("cpu")
  if verbose:
    print(list(predictions.get_fields()))
  indices = [i
            for (i, s) in enumerate(predictions.scores)
            if s >= threshold
            ]

  filtered_predictions = predictions[indices]

  return filtered_predictions


for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])    
    outputs = predictor(im)

    filtered_predictions = filter_predictions_from_outputs(outputs,
                                                           threshold=threshold)
    
    v = Visualizer(im[:, :, ::-1],
                   metadata=custom_metadata, 
                   scale=0.5, 
    )
    out = v.draw_instance_predictions(filtered_predictions)
    #cv2_imshow(out.get_image()[:, :, ::-1])
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(im[:, :, ::-1])  # BGR to RGB
    ax[0].set_title('Original Image ')
    ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
    ax[1].set_title('Detected Image')
    plt.show()


evaluator = COCOEvaluator("custom_val", cfg, False, output_dir=outdir_path)
val_loader = build_detection_test_loader(cfg, "custom_val")
#print(inference_on_dataset(trainer.model, val_loader, evaluator))


def run_worflow(my_image,
                my_model,
                threshold = 0.7,
                verbose=False, filename=None):
  start = time.time()
  outputs = my_model(my_image)
  end = time.time()
  elapsed_time = (end - start) * 1000
  print("Evaluation Time : {} ms ", elapsed_time)

  filtered_predictions = filter_predictions_from_outputs(outputs,
                                                         threshold=threshold,
                                                         verbose=verbose)

  # We can use `Visualizer` to draw the predictions on the image.
  v = Visualizer(my_image[:, :, ::-1], 
                 custom_metadata,
                 scale=1.2)
  out = v.draw_instance_predictions(filtered_predictions)
  #cv2_imshow(out.get_image()[:, :, ::-1])
  fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
  fig.set_dpi(100)
  ax[0].imshow(my_image[:, :, ::-1])  # BGR to RGB
  ax[0].set_title('Original Image ')
  ax[1].imshow(out.get_image()[:, :, ::-1])  # BGR to RGB
  ax[1].set_title('Detected Image')
  if args['save']:
      plt.savefig(filename, dpi=100)
      plt.close(fig)
  else:
      plt.show()

  return

img_name = '../custom_balloon/val2017/3825919971_93fb1ec581_b.jpg'
im = cv2.imread(img_name)
threshold = 0.7
filename1 = "output_images/{}_output1.png".format(args['arch'])
run_worflow(im,
            predictor,
            threshold = threshold,
            filename=filename1)

img_name2 = '../custom_balloon/val2017/16335852991_f55de7958d_k.jpg'
im2 = cv2.imread(img_name2)

filename2 = "output_images/{}_output2.png".format(args['arch'])
run_worflow(im2,
            predictor,
            threshold = threshold,
            filename=filename2)