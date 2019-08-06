## Visualization Tool For Convolutional Neural Networks

![alt text](https://github.com/oliverzhang42/vis/blob/master/saliency.png?raw=true)

Above: Highlighted are the pixels which the resnet_ppg model associates with a 
bad ppg. We use guided saliency as our visualization method.

### How to Use
Currently the tools accepts 9 arguments. The tool can also display and visualize
both 2D and 3D convolutional neural 
networks. 

#### Required Arguments
```--model (str):``` path to neural network
```--unprocessed_img(str):``` path to unprocessed image. (jpg)
```--preprocessed_img (str):``` path to preprocessed img (Must be .npy file)
```--vis (str):``` Determines visualization type. Either 'cam' (heatmap), 
'shap', 'integrated_grad', or 'saliency' 

#### Sometimes Required Arguments
```--background (str):``` Used if you chose 'integrated_grad' or 'shap'. Path 
to the "background image" (If unsure what 
"background image" to use, see note below)
```--conv_layer (int):``` Used if you chose 'cam'. The index of the last conv2d 
layer in the model.

#### Optional Arguments:
```--show (bool):``` Whether to show the image or just save it.
```--contrast (int):``` Used for gradient methods ('shap', 'integrated_grad', 
'saliency'). A number between 1 to 5, the 
higher it is, the more gradients are visible. (More info below)
```--clip (int):``` Used for gradient methods. A number between 1 to 5, clips 
the highest gradient to max/clip, so other
 gradients are visible.
```--neuron (int):``` Which neuron in the last layer to visualize. If left 
blank, the code will visualize the neuron (in
 the last layer) with the highest activation.

#### Notes on arguments:
```--conv_layer``` only comes into play if you're using cam and your last pooling
layer is GlobalAveragPooling. This is the case for ResNet (1D and 2D) but not VGG.

```--background``` only is necessary for shap or integrated_grad. Basically, 
these methods require a base image to compare against. This 'base image' should 
be an image which the model is uncertain about. (For resnet_ppg, a white image 
is a bad background because the model is confident that the PPG is bad.) Also 
note that the background image must be preprocessed like all other images.

```--contrast``` works by modifying the colormap when displaying gradients. This
 means that when contrast isn't one, the colormap isn't a linear mapping.

### How it works

My code is built upon the keras-vis library, the shap library, and the 
Integrated-Gradients library. 

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

(Note: Shap is actually a unified framework of different visualization methods. 
I'm actually using DeepLIFT, a small part of the framework)

1. https://arxiv.org/abs/1704.02685 (DeepLIFT, the subsection of shap that is 
used)
2. https://arxiv.org/abs/1705.07874 (general shap)

integrated_grad is implemented by the Integrated-Gradients library and comes 
from the paper: (Note that installing Integrated-Gradients isn't necessary; I've
 copied and pasted a function over from the library instead.)

1. https://arxiv.org/abs/1703.01365

### Dependencies Required:
```
numpy 1.16.4
matplotlib 3.1.0
keras 2.2.4
keras-vis 0.4.1
shap 0.29.2
```
