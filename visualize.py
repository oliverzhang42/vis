# TODO: Do I want to add preprocessing?
# TODO: Do I want to add SHAP?

import argparse
from keras import activations
from keras.models import load_model
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam
plt.rcParams['figure.figsize'] = (15, 10)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='path of keras model')
parser.add_argument('--unprocessed_img', type=str, help='path of raw image.')
parser.add_argument('--preprocessed_img', type=str, help='path of image. Must be npy file and preprocessed.')
parser.add_argument('--vis', type=str, default='saliency', help='either CAM or saliency')
parser.add_argument('--conv_layer', type=int, default=None,
                    help='Used for CAM. Index of last convolutional layer')

args = parser.parse_args()

assert args.vis in ['cam', 'saliency'], "vis must either be cam or saliency!"

# Load the image
img = np.load(args.preprocessed_img)
unprocessed_img = utils.load_img(args.unprocessed_img)

# Load the model
model = load_model(args.model)

# Make predictions (will be used later)
pred = model.predict(np.array([img]))[0]

# Swap softmax with linear
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)

# Get visualization of last layer
layer_idx = -1
visualizations = []
neurons = model.layers[layer_idx].output_shape[1]

# For each output option, visualize which inputs effect it.
for i in range(neurons):
    if args.vis == 'cam':
        grads = visualize_cam(model, layer_idx, filter_indices=i,
                              seed_input=img, penultimate_layer_idx=args.conv_layer,
                              backprop_modifier=None)
        # Lets overlay the heatmap onto original image.
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        visualizations.append(overlay(jet_heatmap, unprocessed_img, alpha=0.5))
    else:
        visualizations.append(visualize_saliency(model, layer_idx, backprop_modifier='guided',
                                                 filter_indices=i, seed_input=img))

# Plot the visualizations for each output option.
f, ax = plt.subplots(2, neurons)

for i in range(neurons):
    ax[i, 0].set_title("probability: %.3f" % pred[i])
    ax[i, 0].imshow(unprocessed_img)
    ax[i, 1].set_title("neuron: {}".format(i))
    ax[i, 1].imshow(visualizations[i], cmap='jet')

plt.show()