# USAGE train_detr.py --arch <> --path <> --epochs <> --profile True/False
# For example, python3.8 training.py  --arch detr --path model_output --epochs 300

import torchvision
import os
import numpy as np
import os
from PIL import Image, ImageDraw

import pytorch_lightning as pl
from transformers import DetrConfig, DetrForObjectDetection, DeformableDetrForObjectDetection
from transformers import AutoModelForObjectDetection
from transformers import AutoFeatureExtractor
from transformers import YolosFeatureExtractor, YolosForObjectDetection
from ptflops import get_model_complexity_info

import torch
from pytorch_lightning import Trainer
from transformers import DetrFeatureExtractor
import matplotlib.pyplot as plt

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,	help="output path  to the model")
ap.add_argument('-a', '--arch', default='detr', choices=['detr', 'cond-detr', 'yolos', 'def-detr'], help='Choose different transformer based object detection architecture')
ap.add_argument("-e", "--epochs", type=int, help="No of Epochs for training")
ap.add_argument("-r", '--profile', default=False, type=bool, help='Profiling different model')

args = vars(ap.parse_args())

device = None

if args["arch"] == 'def-detr':
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

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


feature_extractor = None

if args["arch"] == 'detr':
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
elif args["arch"] == 'cond-detr':
    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/conditional-detr-resnet-50")
elif args["arch"] == 'yolos':
    feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small", size=512, max_size=864)
elif args["arch"] == 'def-detr':
    feature_extractor = AutoFeatureExtractor.from_pretrained("SenseTime/deformable-detr")

train_dataset = CocoDetection(img_folder='../custom_balloon/train2017', feature_extractor=feature_extractor)
val_dataset = CocoDetection(img_folder='../custom_balloon/val2017', feature_extractor=feature_extractor, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))


# based on https://github.com/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb
image_ids = train_dataset.coco.getImgIds()
# let's pick a random image
image_id = image_ids[np.random.randint(0, len(image_ids))]
print('Image n°{}'.format(image_id))
image = train_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('../custom_balloon/train2017', image['file_name']))

annotations = train_dataset.coco.imgToAnns[image_id]
draw = ImageDraw.Draw(image, "RGBA")

cats = train_dataset.coco.cats
id2label = {k: v['name'] for k,v in cats.items()}

for annotation in annotations:
  box = annotation['bbox']
  class_idx = annotation['category_id']
  x,y,w,h = tuple(box)
  draw.rectangle((x,y,x+w,y+h), outline='red', width=1)
  draw.text((x, y), id2label[class_idx], fill='white')

image.show()

from torch.utils.data import DataLoader

def collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad_and_create_pixel_mask(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['pixel_mask'] = encoding['pixel_mask']
  batch['labels'] = labels
  return batch


def yolos_collate_fn(batch):
  pixel_values = [item[0] for item in batch]
  encoding = feature_extractor.pad(pixel_values, return_tensors="pt")
  labels = [item[1] for item in batch]
  batch = {}
  batch['pixel_values'] = encoding['pixel_values']
  batch['labels'] = labels
  return batch

if args["arch"] == 'yolos':
    train_dataloader = DataLoader(train_dataset, collate_fn=yolos_collate_fn, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=yolos_collate_fn, batch_size=1)
elif args["arch"] == 'def-detr':
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=1)
else :
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=2)

batch = next(iter(train_dataloader))
print(batch.keys())

pixel_values, target = train_dataset[0]

print(pixel_values.shape)
print(target)


class ObjectDetector(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay, architecture):
         super().__init__()
         # replace COCO classification head with custom head
         if architecture == 'detr':
             self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                                 num_labels=len(id2label),
                                                                 ignore_mismatched_sizes=True)
         elif architecture == 'cond-detr':
             self.model = AutoModelForObjectDetection.from_pretrained("microsoft/conditional-detr-resnet-50",
                                                                      id2label={0: "balloon"},
                                                                      label2id={"balloon": 0},
                                                                      ignore_mismatched_sizes=True)
         elif architecture == 'def-detr':
             self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr",
                                                                 id2label={0: "balloon"},
                                                                 label2id={"balloon": 0},
                                                                 ignore_mismatched_sizes=True)
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay

     def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)
        
        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader
     def save_model(self, path):
        self.model.save_pretrained(path)

class YoloS(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained("hustvl/yolos-small",
                                                                 num_labels=len(id2label),
                                                                 ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                      weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def save_model(self, path):
        self.model.save_pretrained(path)

arch = args["arch"]
model = None
output = None

if arch == 'detr' or arch == 'cond-detr' or arch == 'def-detr':
    model = ObjectDetector(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4, architecture=arch)
    outputs = model(pixel_values=batch['pixel_values'], pixel_mask=batch['pixel_mask'])
elif arch == 'yolos':
    model = YoloS(lr=2.5e-5, weight_decay=1e-4)
    outputs = model(pixel_values=batch['pixel_values'])

print("outputs.logits.shape {}".format(outputs.logits.shape))

#trainer = Trainer(gpus=1, max_steps=1, gradient_clip_val=0.1)
trainer = Trainer(accelerator='gpu', devices=1, max_steps=args['epochs'], gradient_clip_val=0.1)
trainer.fit(model)


model_path = "{}".format(args["arch"])
outdir = args["path"]

path = os.path.join(outdir, model_path)
print("path {}".format(path))

if not os.path.exists(path):
    os.makedirs(os.path.join(outdir, model_path))

model.save_model(path)
feature_extractor.save_pretrained(path)
from detr.datasets import get_coco_api_from_dataset

base_ds = get_coco_api_from_dataset(val_dataset) # this is actually just calling the coco attribute

from detr.datasets.coco_eval import CocoEvaluator
from tqdm import tqdm

iou_types = ['bbox']
coco_evaluator = CocoEvaluator(base_ds, iou_types) # initialize evaluator with ground truths

model.to(device)
model.eval()

print("Running evaluation...")

for idx, batch in enumerate(tqdm(val_dataloader)):
    # get the inputs
    pixel_values = batch["pixel_values"].to(device)
    labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized
    if arch == 'detr' or arch == 'cond-detr':
        pixel_mask = batch["pixel_mask"].to(device)
        # forward pass
        outputs = model.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    elif arch == 'yolos':
        outputs = model.model(pixel_values=pixel_values)
    orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
    results = feature_extractor.post_process(outputs, orig_target_sizes) # convert outputs of model to COCO api
    res = {target['image_id'].item(): output for target, output in zip(labels, results)}
    coco_evaluator.update(res)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()


model.to(device)
model.eval()

#We can use the image_id in target to know which image it is
pixel_values, target = val_dataset[1]

pixel_values = pixel_values.unsqueeze(0).to(device)
print("pixel_values.shape: {}".format(pixel_values.shape))

if args["profile"]:
    _, _, width, height = pixel_values.shape
    print("Profiling: Input width = {}, height = {}".format(width, height))
    input = (3, width, height)
    print("=====START Profile With PTFLOPS========")
    macs, params = get_model_complexity_info(model.model, input, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    print("=====END Profile With PTFLOPS========")

if arch == 'detr' or arch == 'cond-detr':
    # forward pass to get class logits and bounding boxes
    outputs = model.model(pixel_values=pixel_values, pixel_mask=None)
elif arch == 'yolos':
    outputs = model.model(pixel_values=pixel_values)


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
  plot_results(pil_img=image, prob=probas[keep], boxes=bboxes_scaled)

#url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#image = Image.open(requests.get(url, stream=True).raw)

image_id = target['image_id'].item()
image = val_dataset.coco.loadImgs(image_id)[0]
image = Image.open(os.path.join('../custom_balloon/val2017', image['file_name']))


if args['arch'] == 'detr' or args['arch'] == 'yolos':
    visualize_predictions(image, outputs)
elif args["arch"] == 'cond-detr':
    # rescale bounding boxes
    target_sizes = torch.tensor(image.size[::-1], device=device).unsqueeze(0)
    results = feature_extractor.post_process(outputs, target_sizes)[0]
    keep = results['scores'] > 0.3
    plot_output(image, results['scores'][keep], results['labels'][keep], results['boxes'][keep])





