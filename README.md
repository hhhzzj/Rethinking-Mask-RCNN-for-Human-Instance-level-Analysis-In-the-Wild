Human pose analysis
-------
We revisit Mask-RCNN and propose a novel Cascaded Mask-RCNN for human instance-level analysis in-the-wild.

Our work is based on [DensePose](https://github.com/facebookresearch/DensePose).

We modify part code of DensePose in ```detectron``` and ```configs```.

![fig](https://github.com/hhhzzj/Cascaded-Mask-RCNN/blob/master/result.png)


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

