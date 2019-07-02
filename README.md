## Visualization Tool For Convolutional Neural Networks

### How to Use
Currently the tools accepts 5 arguments.

```
--model (str): path to neural network
--unprocessed_img (str): path to unprocessed img (jpg or png)
--preprocessed_img (str): path to preprocessed img (Must be .npy file)
--vis (str): Either 'cam' (heatmap), 'shap', 'integrated_grad', or 'saliency' 
--conv_layer (int): Optional argument, used if you chose 'cam'. The index of the last conv2d layer in the model.
--background (str): Optional argument, used if you chose 'integrated_grad' or 'shap'. Path to the "background image"
```

#### Notes on arguments:
```--conv_layer``` only comes into play if your last pooling
layer is GlobalAveragPooling. This is the case for ResNet but not VGG.

```--background``` only is necessary for shap or integrated_grad. Basically, these methods require
a base image to compare against. This 'base image' should be an image which the model is uncertain about.
(For resnet_ppg, a white image is a bad background because the model is confident that the PPG is bad.) 
Also note that the background image must be preprocessed like all other images.

### How it works

My code is built upon the keras-vis library, the shap library, and the Integrated-Gradients library. 

```
keras vis: https://github.com/raghakot/keras-vis
shap: https://github.com/slundberg/shap
Integrated-Gradients: https://github.com/ankurtaly/Integrated-Gradients
```

Saliency is implemented by keras-vis and comes from the papers:

1. https://arxiv.org/abs/1312.6034 (vanilla saliency)
2. https://arxiv.org/abs/1412.6806 (discusses guided saliency p.7)

CAM is implemented by keras-vis and comes from the papers:

1. https://arxiv.org/abs/1512.04150 (basic CAM with limited use case.)
2. https://arxiv.org/abs/1610.02391 (generalization of CAM)

Shap is implemented by the shap library and comes from the papers:

(Note: Shap is actually a unified framework of different visualization methods. I'm actually using DeepLIFT, 
a small part of the framework)

1. https://arxiv.org/abs/1704.02685 (DeepLIFT, the subsection of shap that is used)
2. https://arxiv.org/abs/1705.07874 (general shap)

integrated_grad is implemented by the Integrated-Gradients library and comes from the paper:

1. https://arxiv.org/abs/1703.01365

### Dependencies Required:
```
numpy 1.16.4
matplotlib 3.1.0
keras 2.2.4
keras-vis 0.4.1
shap 0.29.2
```