import argparse
from keras import activations
from keras import backend as K
from keras.models import load_model
from integrated_gradients import integrated_gradients
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
import shap
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam

plt.rcParams['figure.figsize'] = (15, 7)


# TODO: Is name necessary??
def visualize(model, img, unprocessed_img, vis, name, conv_layer=None,
              background=None, show=False, title=None, sensitivity=2,
              neuron=None):
    '''
    Visualizes the predictions of a model on a certain image.

    :param model: (keras model) The model which you want to explain
    :param img: (numpy array) The image the model makes predictions on. Must
    be preprocessed.
    :param unprocessed_img: (numpy array) img but unprocessed. Values either
    ints from 0-255 or floats from 0-1. Will be displayed.
    :param vis: (str) Either "cam", "saliency", "shap" or "integrated_grad".
    The visualization technique
    :param name: (str) The name to save the visualization under
    :param conv_layer: (int) Required if using CAM; the last convolutional
    layer of the model
    :param background: (numpy array) Must be preprocesesd. The "background"
    image for shap or integrated_grad.
    See README for more details.
    :param show: (bool) Whether to display the visualization or just save it.
    :param title: (str) Title of the visualization.
    :param sensitivity: (int) Used with "shap" or "integrated_grad". Sensitivity
    of the visualization.
    Should be between 1-10. %TODO: Should I get rid of sensitivity?
    :param neuron: (int) Which neuron to display. If unset, will display the
    neuron with highest activation.
    :return:
    '''

    # Make predictions (will be used later)
    pred = model.predict(np.array([img]))[0]
    if neuron is None:
        neuron = np.argmax(pred)
    
    # Swap softmax with linear. Makes gradients more visible
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)

    # For each output option, visualize which inputs effect it.
    if vis == 'cam':
        # Get visualization of last layer
        layer_idx = -1

        visualization = visualize_cam(model, layer_idx, filter_indices=neuron,
                              seed_input=img, penultimate_layer_idx=conv_layer,
                              backprop_modifier=None)
    elif vis == 'saliency':
        # Get visualization of last layer
        layer_idx = -1

        visualization = visualize_saliency(model, layer_idx, backprop_modifier='guided',
                                                 filter_indices=neuron, seed_input=img)
    elif vis == 'shap':
        batch = np.array([img])

        # Make sure the variable "background" exists.
        assert 'background' in vars(), "Shap uses a background image"
        # Background needs to be an array of 3D images.
        if len(background.shape) == 3:
            background = np.array([background])
    
        # explain predictions of the model on four images
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(batch)

        # Convert all shap_values to 'float32'
        for i in range(len(shap_values)):
            shap_values[i] = shap_values[i][0].astype('float32')

        # Clip shap_values between min/sensitivity and max/sensitivity
        # We do this to prevent outliers from overshadowing the importances of all other pixels.
        # The higher the sensitivity, the more pixels are at the upper bound, the more lit up the image seems.
        shap_values = np.clip(shap_values, np.min(shap_values)/sensitivity, np.max(shap_values)/sensitivity)

        # Reduce the number of channels of the image. (224, 224, 3) => (224, 224)
        # This is because a 3D importance map is hard to interpret compared to a 2D one.
        shap_values = np.max(shap_values, -1)
    
        # plot the feature attributions
        visualization = shap_values[neuron]
    
    elif vis == 'integrated_grad':
        # Make sure the variable "background" exists
        assert 'background' in vars(), "Integrated Gradients uses a background image"

        # Define a function, given an input and a class index
        # It returns the predictions of your model and the gradients.
        # Such a function is required for the integrated_gradients framework.
        def get_pred_and_grad(input, class_index):
            input = np.array(input)
            pred = model.predict(input)
            fn = K.function([model.input], K.gradients(model.output[:, class_index], model.input))
    
            grad = fn([input])[0]
    
            return pred, grad
    
        attributions = integrated_gradients(
            img,
            neuron,
            get_pred_and_grad,
            baseline=background,
            steps=50
        )
    
        visualization = attributions[0]
    
        # Clip shap_values between min/sensitivity and max/sensitivity
        # We do this to prevent outliers from overshadowing the importances of all other pixels.
        # The higher the sensitivity, the more pixels are at the upper bound, the more lit up the image seems
        visualization = np.clip(visualization, np.min(visualization)/sensitivity, np.max(visualization)/sensitivity)

        # Reduce the number of channels of the image. (224, 224, 3) => (224, 224)
        # This is because a 3D importance map is hard to interpret compared to a 2D one.
        visualization = np.max(visualization, -1)

    # Plot the visualizations for each output option.
    f, ax = plt.subplots(1, 3)

    if title:
        f.suptitle(title)

    # Normalize the visualization to the 0-255 range
    normalized = visualization - np.min(visualization)
    normalized = normalized / np.max(normalized)

    # Lets overlay the heatmap onto original image.
    viridis_heatmap = np.uint8(cm.viridis(normalized)[..., :3] * 255)
    overlaid = overlay(viridis_heatmap, unprocessed_img, alpha=0.5)

    # First Subplot
    ax[0].set_title("Neuron {} activation: %.6f".format(neuron) % pred[neuron])
    ax[0].imshow(unprocessed_img)

    # Second subplot
    ax[1].set_title("Importance map for neuron {}:".format(neuron))
    heatmap = ax[1].imshow(visualization, cmap='viridis')
    f.colorbar(heatmap, ax=ax[1], cmap='viridis')

    # Third subplot
    ax[2].set_title("Overlaid Image")
    overlaid_subplot = ax[2].imshow(overlaid)
    f.colorbar(overlaid_subplot, ax=ax[2], cmap='viridis')

    print("Saving Figure to {}_explained_{}.png".format(name, vis))
    plt.savefig('{}_explained_{}.png'.format(name, vis))
    
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, help='path of keras model')
    parser.add_argument('--unprocessed_img', type=str, help='path of raw image.')
    parser.add_argument('--preprocessed_img', type=str, help='path of image. Must be npy file and preprocessed.')
    parser.add_argument('--vis', type=str, default='saliency', help='either CAM or saliency')
    parser.add_argument('--conv_layer', type=int, default=None,
                        help='Used for CAM. Index of last convolutional layer')
    parser.add_argument('--background', type=str, help='path of "background" image')
    parser.add_argument('--show', type=bool, default=False, help='whether to display the image')
    parser.add_argument('--sensitivity', type=int, default=2, help='Used for shap or integrated_grad. '
                                                                   'Sensitvity of display.')
    parser.add_argument('--neuron', type=int, default=None, help='Which neuron to visualize. If blank, will visualize '
                                                                 'the neuron with highest activation')
    # TODO: There seems to be a bug with show always showing.

    args = parser.parse_args()

    assert args.vis in ['cam', 'saliency', 'shap', 'integrated_grad'], \
        "vis must be cam, shap, integrated_grad or saliency!"

    # Load the image
    preprocessed_img = np.load(args.preprocessed_img)
    unprocessed_img = utils.load_img(args.unprocessed_img)

    # Load the background if there is one
    # Note: Background must be preprocessed!
    if args.background:
        background = np.load(args.background)
    else:
        background = None

    # Load the model
    model = load_model(args.model)

    visualize(model, preprocessed_img, unprocessed_img, args.vis, args.unprocessed_img[:-4],
              conv_layer=args.conv_layer, background=background, show=args.show, sensitivity=args.sensitivity,
              neuron=args.neuron)
