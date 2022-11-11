import torchvision
import os
import numpy as np
import os
from PIL import Image, ImageDraw

import pytorch_lightning as pl
from transformers import DetrForObjectDetection, DetrFeatureExtractor, DeformableDetrForObjectDetection
import torch
from transformers import DetrFeatureExtractor
import matplotlib.pyplot as plt
from transformers import AutoModelForObjectDetection, AutoFeatureExtractor


import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,	help="path to the model")
ap.add_argument('-a', '--arch', default='detr', choices=['detr', 'cond-detr', 'yolos', 'def-detr'], help='Choose different transformer based object detection architecture')
args = vars(ap.parse_args())

device = torch.device("cpu")

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, feature_extractor, train=True):
        ann_file = os.path.join(img_folder, "custom_train.json" if train else "custom_val.json")
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.feature_extractor = feature_extractor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super(CocoDetection, self).__getitem__(idx)
        
        # preprocess image and target (converting target to DETR format, resizing + normalization of both image and target)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        target = encoding["labels"][0] # remove batch dimension

        return pixel_values, target

model = None
feature_extractor = None

model_path = args["model"]
# load best saved model checkpoint from the current run
if os.path.exists(model_path):
    if args["arch"] == 'detr':
        feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained(model_path)

    elif args["arch"] == 'cond-detr':
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/conditional-detr-resnet-50")
        model = AutoModelForObjectDetection.from_pretrained(model_path)
    elif args["arch"] == 'yolos':
        feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=512, max_size=864)
        model = AutoModelForObjectDetection.from_pretrained(model_path)
    elif args["arch"] == 'def-detr':
        feature_extractor = AutoFeatureExtractor.from_pretrained("SenseTime/deformable-detr")
        model = DeformableDetrForObjectDetection.from_pretrained(model_path)
    print('Loaded {} model from this run.'.format(model_path))

train_dataset = CocoDetection(img_folder='balloon/train', feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder='balloon/val', feature_extractor=feature_extractor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))


cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}


#We can use the image_id in target to know which image it is
pixel_values, target = val_dataset[1]

pixel_values = pixel_values.unsqueeze(0).to(device)
print(pixel_values.shape)

if args["arch"] == 'detr' or args["arch"] == 'cond-detr' or args["arch"] == 'def-detr':
    # forward pass to get class logits and bounding boxes
    outputs = model(pixel_values=pixel_values, pixel_mask=None)
elif args["arch"] == 'yolos':
    outputs = model(pixel_values=pixel_values)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_output(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def visualize_predictions(image, outputs, threshold=0.9):
  # keep only predictions with confidence >= threshold
  probas = outputs.logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > threshold
  
  # convert predicted boxes from [0; 1] to image scales
  bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

  # plot results
  plot_results(image, probas[keep], bboxes_scaled)

image_id = target['image_id'].item()
image = val_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('balloon/val', image['file_name']))

if args['arch'] == 'detr' or args['arch'] == 'yolos' :
    visualize_predictions(image, outputs)
elif args["arch"] == 'cond-detr' or args['arch'] == 'def-detr':
    # rescale bounding boxes
    target_sizes = torch.tensor(image.size[::-1], device=device).unsqueeze(0)
    results = feature_extractor.post_process(outputs, target_sizes)[0]
    keep = results['scores'] > 0.3
    plot_output(image, results['scores'][keep], results['labels'][keep], results['boxes'][keep])

