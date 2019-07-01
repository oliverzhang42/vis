import argparse
from keras import activations
from keras.models import load_model
from integrated_grad_visualize import integrated_gradients
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam

from keras import backend as K
import shap

plt.rcParams['figure.figsize'] = (15, 10)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('model', type=str, help='path of keras model')
parser.add_argument('unprocessed_img', type=str, help='path of raw image.')
parser.add_argument('preprocessed_img', type=str, help='path of image. Must be npy file and preprocessed.')
parser.add_argument('vis', type=str, default='saliency', help='either CAM or saliency')
parser.add_argument('--conv_layer', type=int, default=None,
                    help='Used for CAM. Index of last convolutional layer')
parser.add_argument('--background', type=str, help='path of "background" image')
parser.add_argument('--show', type=bool, default=True, help='whether to display the image')
# TODO: There seems to be a bug with show always showing.

args = parser.parse_args()

assert args.vis in ['cam', 'saliency', 'shap', 'integrated_grad'], \
    "vis must be cam, shap, integrated_grad or saliency!"

# Load the image
img = np.load(args.preprocessed_img)
batch = np.array([img])
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

if args.vis == 'cam':
    for i in range(neurons):
        grads = visualize_cam(model, layer_idx, filter_indices=i,
                              seed_input=img, penultimate_layer_idx=args.conv_layer,
                              backprop_modifier=None)
        # Lets overlay the heatmap onto original image.
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        visualizations.append(overlay(jet_heatmap, unprocessed_img, alpha=0.5))
elif args.vis == 'saliency':
    for i in range(neurons):
        visualizations.append(visualize_saliency(model, layer_idx, backprop_modifier='guided',
                                                 filter_indices=i, seed_input=img))
elif args.vis == 'shap':
    # select a set of background examples to take an expectation over
    # NOTE: Not sure what background image to use, an expectation or a white image.
    # Trying white image for now.
    assert args.background, "Shap uses a background image"
    background = np.load(args.background)
    if len(background.shape) == 3:
        background = np.array([background])

    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)

    shap_values = e.shap_values(batch)

    shap_values[0] = shap_values[0][0].astype('float32')
    shap_values[1] = shap_values[1][0].astype('float32')

    # normalize the arrays
    shap_values = shap_values / (np.max(shap_values) - np.min(shap_values))

    # plot the feature attributions
    visualizations = shap_values

elif args.vis == 'integrated_grad':
    for i in range(neurons):
        assert args.background, "Integrated Gradients uses a background image"
        background = np.load(args.background)

        def get_pred_and_grad(input, class_index):
            input = np.array(input)
            pred = model.predict(input)
            fn = K.function([model.input], K.gradients(model.output[:, class_index], model.input))

            grad = fn([input])[0]

            return pred, grad


        attributions = integrated_gradients(
            img,
            i,
            get_pred_and_grad,
            baseline=background,
            steps=50
        )

        grad = attributions[0]

        # removing all negative gradients
        grad = np.clip(grad, 0, np.max(grad))
        grad = np.max(grad, -1)

        visualizations.append(grad)

# Plot the visualizations for each output option.
f, ax = plt.subplots(2, neurons)

for i in range(neurons):
    ax[i, 0].set_title("Neuron {} activation: %.6f".format(i) % pred[i])
    ax[i, 0].imshow(unprocessed_img)
    ax[i, 1].set_title("Importance map for neuron {}:".format(i))
    heatmap = ax[i, 1].imshow(visualizations[i])
    f.colorbar(heatmap, ax=ax[i, 1])

print("Saving Figure to " + args.unprocessed_img[:-4] + '_explained.png')
plt.savefig(args.unprocessed_img[:-4] + '_explained.png')

if args.show:
    plt.show()