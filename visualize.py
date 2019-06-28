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

from keras import backend as K
import shap

plt.rcParams['figure.figsize'] = (15, 10)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='path of keras model')
parser.add_argument('--unprocessed_img', type=str, help='path of raw image.')
parser.add_argument('--preprocessed_img', type=str, help='path of image. Must be npy file and preprocessed.')
parser.add_argument('--vis', type=str, default='saliency', help='either CAM or saliency')
parser.add_argument('--conv_layer', type=int, default=None,
                    help='Used for CAM. Index of last convolutional layer')
parser.add_argument('--background', type=str, help='path of "background" image')

args = parser.parse_args()

assert args.vis in ['cam', 'saliency', 'shap'], "vis must be cam, shap, or saliency!"

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
    elif args.vis == 'saliency':
        visualizations.append(visualize_saliency(model, layer_idx, backprop_modifier='guided',
                                                 filter_indices=i, seed_input=img))
    elif args.vis == 'shap':

        #array = np.array([img])
        #'''

        import pudb; pudb.set_trace()
        img = np.array([img]).astype('float32')

        def map2layer(x, layer):
            feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
            return K.get_session().run(model.layers[layer].input, feed_dict)

        e = shap.GradientExplainer(
            (model.layers[15].input, model.layers[-1].output),
            map2layer(img, 15),
            local_smoothing=0  # std dev of smoothing noise
        )
        shap_values,indexes = e.shap_values(map2layer(img, 15), ranked_outputs=1)

        # plot the explanations
        shap.image_plot(shap_values, img)
        '''

        # select a set of background examples to take an expectation over
        # NOTE: Not sure what background image to use, an expectation or a white image.
        # Trying white image for now.
        background = np.ones((1, 224, 224, 3))  # x_train[np.random.choice(x_train.shape[0], 100, replace=False)]

        #background = np.mean(np.load("good_ppg.npy"), 0)
        #background = np.array([background])
        #import pudb; pudb.set_trace()

        # explain predictions of the model on four images
        e = shap.DeepExplainer(model, background)
        # ...or pass tensors directly
        # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
        shap_values = e.shap_values(array)

        shap_values[0] = shap_values[0].astype('float32')
        shap_values[1] = shap_values[1].astype('float32')

        # plot the feature attributions
        shap.image_plot(shap_values, -array.astype('float32'))
        #'''

# Plot the visualizations for each output option.
f, ax = plt.subplots(2, neurons)

for i in range(neurons): #TODO: Change this disgusting formatting lol
    ax[i, 0].set_title("Neuron {} activation: %.6f".format(i) % pred[i])
    ax[i, 0].imshow(unprocessed_img)
    ax[i, 1].set_title("Importance map for neuron {}:".format(i))
    ax[i, 1].imshow(visualizations[i], cmap='jet')

plt.savefig(args.unprocessed_img[:-4] + '_explained.png')

#plt.show() #TODO: Make it automatically save the image. Give an option to not display it
# TODO: Perhaps create an option to automatically explain all images in a folder?