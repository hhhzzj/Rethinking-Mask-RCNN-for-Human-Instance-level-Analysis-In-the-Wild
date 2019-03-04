We revisit Mask-RCNN and propose a novel Cascaded Mask-RCNN for human instance-level analysis in-the-wild.

![fig](https://github.com/hhhzzj/Cascaded-Mask-RCNN/blob/master/result.png)


Training a model
-------
This example shows how to trian a model on the DensePose-COCO dataset. We can choose different structure to train using different config
files. The model will be an end-to-end trained DensePose-RCNN using a ResNet-50-FPN backbone.

```
python2 tools/train_net.py \
    --cfg configs/DensePose_ResNet50_FPN_single_GPU.yaml \
    OUTPUT_DIR /tmp/detectron-output
```

Testing a pretrianed model
-------
Before testing, you should make sure that you have downloaded the pretrianed model. This example shows how to run an end-to-end trained DensePose-RCNN model from the model zoo using a single GPU for inference. 
```
python2 tools/test_net.py \
    --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml \
    TEST.WEIGHTS https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl \
    NUM_GPUS 1
```

