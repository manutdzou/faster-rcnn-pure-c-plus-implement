# faster-rcnn-pure-c plus implement
c++ proposal layer and c++ wrapper

This project is based on the official caffe implement. In this project we transform the [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) to pure C++ code style.

The proposal layer in faster rcnn is rewritten in C++, and the test wrapper is transformed to C++ too.

## Add the Roipooling layer and Smooth_L1_loss_layer

copy the roi_pooling_layer.hpp and smooth_L1_loss_layer.hpp from the [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) to the [official caffe branch](http://caffe.berkeleyvision.org/) caffe/include/caffe/layers/, then copy the roi_pooling_layer.c* and smooth_L1_loss_layer.c* to caffe/src/caffe/layers/

add the proto information of roi_pooling_layer and smooth_L1_loss_layer like

```
  optional ROIPoolingParameter roi_pooling_param = 200;
  optional SmoothL1LossParameter smooth_l1_loss_param = 201;


// Message that stores parameters used by ROIPoolingLayer
message ROIPoolingParameter {
  // Pad, kernel size, and stride are all given as a single value for equal
  // dimensions in height and width or as Y, X pairs.
  optional uint32 pooled_h = 1 [default = 0]; // The pooled output height
  optional uint32 pooled_w = 2 [default = 0]; // The pooled output width
  // Multiplicative spatial scale factor to translate ROI coords from their
  // input scale to the scale used when pooling
  optional float spatial_scale = 3 [default = 1];
}
message SmoothL1LossParameter {
  // SmoothL1Loss(x) =
  //   0.5 * (sigma * x) ** 2    -- if x < 1.0 / sigma / sigma
  //   |x| - 0.5 / sigma / sigma -- otherwise
  optional float sigma = 1 [default = 1];
}
```

## Then add the proposal layer with C++ implement and proto information like

```
  optional ProposalParameter proposal_param = 202;
  
  
message ProposalParameter {
  optional int32 feat_stride = 1 [default = 16];
  optional int32 anchor_base_size = 2 [default = 16];
  optional int32 anchor_scale = 3 [default = 3];
  optional int32 anchor_ratio = 4 [default = 3];
  optional int32 max_rois = 5 [default = 300];
}
```

## Next modify the test prototxt of proposal layer to C++ layer style

```
layer {
  name: 'proposal'
  type: 'Proposal'
  bottom: 'rpn_cls_prob_reshape'
  bottom: 'rpn_bbox_pred'
  bottom: 'im_info'
  top: 'rois'
  proposal_param {
   feat_stride: 16;
   anchor_base_size: 16;
   anchor_scale: 3;
   anchor_ratio: 3;
   max_rois: 300;
  }
}
```
