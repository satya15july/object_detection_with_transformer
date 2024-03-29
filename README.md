# Introduction
This project try to use Detectron2 and Huggingface framework to do object detection using transformer based architecture.


# Object Detection with Trasformer using Huggingface

With Huggingface framework, the following transformer based architectures are used

-   DETR(Detection Transformer).
-   Conditional DETR
-   Deformable DETR.
-   YOLOS.

## Dependencies
- Install Huggingface by following steps mentioned in [link](https://huggingface.co/docs/transformers/installation).
- pip install pytorch-lightning


## Dataset Preparation
  Ballon Dataset is converted to COCO format & present inside custom_balloon folder.

## Usage


### Training:

Currently Huggingface only supports following trasformer based object detection algorithm:

-   DETR
-   Conditional DETR
-   Deformable DETR
-   YOLOS

Run the below command for training

-   python3.8 training.py  --arch [detr|cond-detr|yolos|def-detr] --path model_output/[detr|cond-detr|yolos|def-detr] --epochs 5000 --profile True.

         --path model_output:  Use different folder for different architecture.

### Profiling

![obj-detec-params](https://user-images.githubusercontent.com/22910010/213909576-2b09d2fb-9dcd-4a9f-875f-a8253def6011.png)


### Inference:

Run the below command

python3.8 inference.py --model model_out/detr  --arch detr/cond-detr/yolos/def-detr

Evaluation time with different model is as follows:

-   Evaluation Time for arch: detr is 762.916088104248 ms.
-   Evaluation Time for arch: yolos is 384.78732109069824 ms.
-   Evaluation Time for arch: cond-detr is 776.5250205993652 ms.
-   Evaluation Time for arch: def-detr is 2585.845708847046 ms.

### Output:

#### Original Image:

![original](https://user-images.githubusercontent.com/22910010/213909599-60b54893-42f0-40a4-b5e7-62a51965f971.jpg)


#### DETR Output:

![output_detr](https://user-images.githubusercontent.com/22910010/213909671-04981745-273f-4fc0-8547-1d675f2579a9.png)


#### Cond-DETR Output:

![output_cond-detr](https://user-images.githubusercontent.com/22910010/213909688-40481c83-a1e5-4e11-bca1-40ee79e33fcf.png)


#### Deformable DETR Output:

![output_def-detr](https://user-images.githubusercontent.com/22910010/213909708-350e173c-460a-4968-8184-56445846b8b2.png)


#### YOLOS Output:

![output_yolos](https://user-images.githubusercontent.com/22910010/213909737-f50e6476-e230-4df5-9f4e-40cdd4426dbc.png)


## Summary
In the original image, only 7 balloons are present and it was detected correctly with Cond-Detr & Def-Detr.

Detr model able to predict only 6 balloons & misses 1 prediction.Yolos is able to predict only 5 balloons &

misses 2 predictions.However,Yolos is the fastest architecture among all,whereas Def-Detr takes longer time 
than others.(Note: All the models were trained for 500 epochs).

So there is a clear trade-off between accuracy & speed.Please check the profiling data mentioned above.

Accuracy can be improved by finetuning the hyper parameters or with more training.

But the clear winner in terms of speed is Yolos & in terms of accuracy it's Cond-Detr & Def-Detr.

# Object Detection with Trasformer Using Detectron2

## Dependencies
  1. Install detectron2.[](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) (Prefer to use Conda version).
  2. Install DyHead by following the steps present in [DynamicHead](https://github.com/microsoft/DynamicHead).

     You may face some building issue related to CUDA in DynamicHead/dyhead/csrc/cuda/{deform_conv_cuda.cu, SigmoidFocalLoss_cuda.cu}.
     Try to Fix them.Otherwise,let me know what is the error you are facing.

## Dataset Preparation
  Balloon dataset is converted to COCO format & is present inside custom_balloon folder.

  If you want to convert balloon dataset in to coco format & use it in Detectron2.Then,
  follow the below steps.

    -   Download the balloon dataset from https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip.
    -   git clone https://github.com/woctezuma/VIA2COCO
    -   cd VIA2COCO/
        git checkout fixes
    -   run convert_coco.py


## Usage

### Enviornment Setup
-   For DyHead:
    1. Download weights(for example, dyhead_r50_atss_fpn_1x.pth) from [DynamicHead](https://github.com/microsoft/DynamicHead) & keep them inside pretrained_model/
    2. Copy the config files from [](https://github.com/microsoft/DynamicHead/tree/master/configs) & keep them inside configs/

-   For DETR:
    1. git clone https://github.com/facebookresearch/detr/tree/master/d2.
    2. Keep the converted_model.pth inside pretrained_model/ 
       Here is the steps to get the converted_model.pth
           - python converter.py --source_model https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --output_model converted_model.pth
    3. Copy the config file https://github.com/facebookresearch/detr/tree/master/d2/configs/detr_256_6_6_torchvision.yaml & keep it inside configs/ 

### Training
- For DyHead with FPN backbone:

      python3.8 training.py --outdir [where model will be saved] --arch dyhead-fpn --config [file path] --weight [file path] --epochs [no of epochs].

      For Example,
          python3.8 training.py --outdir out_dyhead_fpn/ --arch dyhead-fpn --config configs/dyhead_r50_atss_fpn_1x.yaml --weight pretrained_model/dyhead_r50_atss_fpn_1x.pth --epochs 5000

- For DyHead with Swin-T transformer backbone:

      python3.8 training.py --outdir [where model will be saved] --arch dyhead-swint --config [file path] --weight [file path] --epochs [no of epochs]

      For Example,
          python3.8 training.py --outdir out_dyhead_swint/ --arch dyhead-swint --config configs/dyhead_swint_atss_fpn_2x_ms.yaml --weight pretrained_model/dyhead_swint_atss_fpn_2x_ms.pth --epochs 5000.

- For DETR:

      python3.8 training.py --outdir [where model will be saved] --arch detr --config [file path] --weight [file path] --epochs [no of epochs].

      For Example,
          python3.8 training.py --outdir out_test/ --arch detr --config configs/detr_256_6_6_torchvision.yaml --weight pretrained_model/converted_model.pth --epochs 5000

### Inference:

- For DyHead with FPN backbone:

    python3.8 inference.py --outdir out_dyhead_fpn/ --arch dyhead-fpn --config configs/dyhead_r50_atss_fpn_1x.yaml --save True.

    Inference Time:

       Evaluation Time : {} ms  108.9015007019043
       Evaluation Time : {} ms  103.93381118774414

    ![dyhead_output2](https://user-images.githubusercontent.com/22910010/213909844-eb137c71-5012-4115-aea5-f0ad0b05c1d0.png)

- For DyHead with Swin-T transformer backbone:
   python3.8 inference.py --outdir out_dyhead_swint/ --arch dyhead-swint --config configs/dyhead_swint_atss_fpn_2x_ms.yaml --save True.

   Inference Time:
      Evaluation Time : {} ms  157.5005054473877.
      Evaluation Time : {} ms  153.02109718322754

- For DETR:

    python3.8 inference.py --outdir out_detr/ --arch detr --config configs/detr_256_6_6_torchvision.yaml --save True.

    Inference Time:

        Evaluation Time : {} ms  71.02847099304199
        Evaluation Time : {} ms  92.53978729248047


![detr_output2](https://user-images.githubusercontent.com/22910010/213909800-cf7d1edc-a395-438e-ad6c-a58895999e80.png)


## Summary:
As you can see from output, DETR is slighly faster than DyHead.However,DETR is not that accurate as DyHead in predicting all the ballons.

Please check the above output.

We can try other DyHead configs such as dyhead_swint_atss_fpn_2x_ms.yaml and check the output.

Here the idea is to demonstrate how to use trasformer based object detection using Detectron2 framework.
Please feel free to share your feedback.

## References:

-   https://huggingface.co/docs/transformers/tasks/image_classification.
-   https://github.com/NielsRogge/Transformers-Tutorials.
-   https://arxiv.org/pdf/2005.12872.
-   https://arxiv.org/abs/2010.04159.
-   https://arxiv.org/pdf/2108.06152.
-   https://arxiv.org/pdf/2106.00666.
-   https://arxiv.org/pdf/2106.08322.
-   https://arxiv.org/abs/2202.09048.
-   https://arxiv.org/abs/2111.14330.
-   https://github.com/facebookresearch/detr/tree/master/d2.
-   https://github.com/microsoft/DynamicHead.

---
Reach me @

[LinkedIn](https://www.linkedin.com/in/satya1507/) [GitHub](https://github.com/satya15july) [Medium](https://medium.com/@satya15july_11937)
