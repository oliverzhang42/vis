import argparse
from keras import activations
from keras.models import load_model
import keras.backend as K
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
import shap
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam
plt.rcParams['figure.figsize'] = (15, 10)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='path of keras model')
parser.add_argument('--image', type=str, help='path of image')
parser.add_argument('--backprop', type=str, default='standard', help='relu, standard, or guided')
parser.add_argument('--cam', type=bool, default=False, help='if you want it to use CAM')
parser.add_argument('--shap', type=bool, default=False, help='if you want to use SHAP')

args = parser.parse_args()

assert args.backprop in ['relu', 'standard', 'guided']

import pudb; pudb.set_trace()

# ======================================================

# Load the model
model = load_model(args.model)

# Load the image
img = utils.load_img(args.image, target_size=model.input_shape[1:3])

# Preprocess???


# Make predictions (will be used later)
pred = model.predict(np.array([img]))[0]

if args.shap:
    img = np.array([img])

    def map2layer(x, layer):
        feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
        return K.get_session().run(model.layers[layer].input, feed_dict)

    e = shap.GradientExplainer(
        (model.layers[15].input, model.layers[-1].output),
        map2layer(img, 15),
        local_smoothing=0  # std dev of smoothing noise
    )
    shap_values,indexes = e.shap_values(map2layer(img, 15), ranked_outputs=1)

    # get the names for the classes
    index_names = "hi"

    # plot the explanations
    shap.image_plot(shap_values, img, index_names)
else:
    # Swap softmax with linear
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)

    # ======================================================

    #'''
    # Get visualization of last layer
    layer_idx = -1
    visualizations = []
    neurons = model.layers[layer_idx].output_shape[1]

    #'''
    for i in range(neurons):
        if args.cam:
            grads = visualize_cam(model, layer_idx, filter_indices=i,
                                  seed_input=img, penultimate_layer_idx=None,
                                  backprop_modifier=None)
            # Lets overlay the heatmap onto original image.
            jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
            visualizations.append(overlay(jet_heatmap, img))
        else:
            if args.backprop == 'standard':
                args.backprop = None
            visualizations.append(visualize_saliency(model, layer_idx, backprop_modifier=args.backprop,
                                                     filter_indices=i, seed_input=img))
    #'''

    # Plot the layers

    f, ax = plt.subplots(2, neurons)

    for i in range(neurons):
        ax[i, 0].set_title("probability: {}".format(pred[i]))
        ax[i, 0].imshow(img)
        ax[i, 1].set_title("neuron: {}".format(i))
        ax[i, 1].imshow(visualizations[i], cmap='jet')

    plt.show()