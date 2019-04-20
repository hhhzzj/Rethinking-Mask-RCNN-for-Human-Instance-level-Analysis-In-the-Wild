Rethinking Mask-RCNN for Human Instance-level Analysis In-the-Wild
-------
![fig](https://github.com/hhhzzj/Rethinking-Mask-RCNN-for-Human-Instance-level-Analysis-In-the-Wild/blob/master/result_5.gif)
![fig](https://github.com/hhhzzj/Rethinking-Mask-RCNN-for-Human-Instance-level-Analysis-In-the-Wild/blob/master/result_7.gif)
Mask-RCNN is a flexible and powerful tool by explicitly
extending the Faster-RCNN with RoIAlign
to produce object bounding-boxes and their corresponding
instance-level object segmentations. Despite
of its success, a severe problem faced by
Mask-RCNN is that the generated bounding-box
often contains multi-human instances in the presence
of partially occlusions, overlapping and scales
of variations. This incurs considerable performance
drop of Mask-RCNN. Also, Mask-RCNN
cannot directly handle (2D)image-to-(3D)surface
estimation. To address these issues, in this paper,
we revisit Mask-RCNN and propose a novel
Cascaded Mask-RCNN for human instance-level
analysis in-the-wild. It enables the estimation
of the 2D image to a human body 3D surface
(image-to-surface) based representations (i.e.,
Index-to-Patch (I), and UV coordinates) and also
elegantly handles multi-instance in one bounding
box problem.Specifically, we design a semanticto-
instance branch by involving intermediate supervision
to effectively segment multi-instances in
one bounding box, and we utilize an IUV-RCNN
branch to simultaneously predict 3D surface corresponding
representations. Extensive experiments
on the large-scale and challenging dataset (i.e.,
DensePose-COCO) demonstrate the effectiveness
of our proposed method. For DensePose task, our
method significantly surpasses the state-of-the-art
by 10.7% in UV AP and 9.2% average in UV
AR. Codes and models are publicly available at
https://github.com/hhhzzj/Cascaded-Mask-RCNN.

![fig](https://github.com/hhhzzj/Cascaded-Mask-RCNN/blob/master/result.png)

Figure 1: Our Cascaded Mask-RCNN prediction results. It aims to estimate dense correspondences from a 2D image in the wild to a 3D
surface-based presentations (i.e., Index-to-Patch, and specific U and V coordinates) of a human body. Each example is arranged from left to
right with the following order: an input image and its corresponding 3D Index-to Patch, U coordinates and V coordinates.


Training a model
-------
This example shows how to trian a model on the DensePose-COCO dataset. We can use different structure to train using different config. The model uses a ResNet-50-FPN backbone with an end-to-end trianing approach.

```
python2 tools/train_net.py \
    --cfg configs/coco_exp_configs/DensePose_ResNet50_FPN_cascade_mask_dp_s1x-e2e.yaml \
    OUTPUT_DIR /tmp/detectron-output
```

Testing a pretrianed model
-------
Before testing, you should make sure that you have downloaded the pretrianed model. This example shows how to run a pretrained model using a single GPU for inference. 
```
python2 tools/test_net.py \
    --cfg configs/coco_exp_configs/DensePose_ResNet50_FPN_cascade_mask_dp_s1x-e2e.yaml \
    TEST.WEIGHTS /the/dir/of/your/trained/model \
    NUM_GPUS 1
```
Qualitative Results
-------
Compare with Mask RCNN in terms of instance segmentation.
![fig](https://github.com/hhhzzj/Rethinking-Mask-RCNN-for-Human-Instance-level-Analysis-In-the-Wild/blob/master/compare.png)

Mask RCNN (in the left) and our result (in the right).
