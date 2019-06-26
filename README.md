## Visualization Tool For Convolutional Neural Networks

### How to Use
Currently the tools accepts 5 arguments.

```
--model (str): path to neural network
--unprocessed_img (str): path to unprocessed img (jpg or png)
--preprocessed_img (str): path to preprocessed img (Must be .npy file)
--vis (str): Either 'cam' (heatmap) or 'saliency' 
--conv_layer (int): Optional argument, used if you chose 'cam'. The index of the last conv2d layer in the model.
```

Note that --conv_layer only comes into play if your last pooling
layer is GlobalAveragPooling. This is the case for ResNet but not VGG.

### How it works

All my code is built upon https://github.com/raghakot/keras-vis. 

Saliency comes from the papers:

1. https://arxiv.org/abs/1312.6034 (vanilla saliency)
2. https://arxiv.org/abs/1412.6806 (discusses guided saliency p.7)

CAM comes from the papers:

1. https://arxiv.org/abs/1512.04150 (basic CAM with limited use case.)
2. https://arxiv.org/abs/1610.02391 (generalization of CAM)