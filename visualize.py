import argparse
from keras import activations
from keras import backend as K
from keras.models import load_model
from integrated_gradients import integrated_gradients
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import shap
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam

plt.rcParams['figure.figsize'] = (15, 7)


def normalize(x):
    '''Noramlize array x to 0-1'''
    normalized = x - np.min(x)
    normalized = normalized / np.max(normalized)
    return normalized


def rgb2gray(rgb):
    '''Converts (n,m,3) rgb image to (n,m) grayscale image'''
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def visualize_integrated_gradients(model, img, background, clip, neuron):
    # Define a function, given an input and a class index
    # It returns the predictions of your model and the gradients.
    # Such a function is required for the integrated_gradients framework.
    def get_pred_and_grad(input, class_index):
        input = np.array(input)
        pred = model.predict(input)
        fn = K.function([model.input],
                        K.gradients(model.output[:, class_index], model.input))

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

    # Clip shap_values between 0 and max/clip
    # Prevents outliers from overshadowing the importances of all other pixels.
    # The higher the clip, the more pixels are at the upper bound, the
    # more lit up the image seems.
    visualization = np.clip(visualization, 0, np.max(visualization) / clip)

    # Reduce the number of channels of the image. (224, 224, 3) => (224, 224)
    # This is because a 3D importance map is hard to interpret compared to a 2D one.
    visualization = np.max(visualization, -1)

    return visualization


def visualize_shap(model, img, background, clip, neuron):
    batch = np.array([img])
    background = np.array([background])

    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(batch)

    # Convert all shap_values to 'float32'
    for i in range(len(shap_values)):
        shap_values[i] = shap_values[i][0].astype('float32')

    # Clip shap_values between 0 and max/clip
    # Prevents outliers from overshadowing the importances of all other pixels.
    # The higher the clip, the more pixels are at the upper bound, the
    # more lit up the image seems.
    shap_values = np.clip(shap_values, 0, np.max(shap_values) / clip)

    # Reduce the number of channels of the image. (224, 224, 3) => (224, 224)
    # This is because a 3D importance map is hard to interpret compared to a 2D one.
    shap_values = np.max(shap_values, -1)

    # plot the feature attributions
    visualization = shap_values[neuron]

    return visualization


def display_2d(visualization, img, neuron, pred, title):
    '''
    Displays a multi-colored line which is actually many uniformly colored
    small line segments.

    Based off of this tutorial:
    https://matplotlib.org/gallery/lines_bars_and_markers/multicolored_line.html
    '''
    f, ax = plt.subplots()

    if title:
        f.suptitle(title)

    y = img
    x = [[i] for i in range(len(y))]

    # We have points in a (7201, 1, 2) shape. In general, we have (num points, 1, 2)
    points = np.concatenate((x, y), 1)
    points = np.expand_dims(points, 1)

    # We turn these points into an array of line segments with shape (7201, 2, 2)
    # segments[0], for example, might be ([2, 3], [4, 5]) which is the line from (2,3) to (4,5)
    segments = np.concatenate((points[:-1], points[1:]), 1)

    norm = plt.Normalize(visualization.min(), visualization.max())
    lc = matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)

    # This determines the colorings
    lc.set_array(visualization)
    lc.set_linewidth(2)

    ax.set_title("Neuron {} activation: %.6f".format(neuron) % pred[neuron])

    heatmap = ax.add_collection(lc)
    ax.set_xlim(np.min(x), np.max(x))
    f.colorbar(heatmap, ax=ax)


def display_3d(visualization, unprocessed_img, neuron, pred, title, vis, contrast):
    '''
    First grayscales and fades the unprocessed_img
    Second converts "visualization" into a displayable image. Overlays \
    visualization on the unprocessed_img
    Third displays everything.
    '''
    f, ax = plt.subplots(1, 3)

    if title:
        f.suptitle(title)

    # Normalize the visualization to the 0-1 range
    normalized = normalize(visualization)

    # Grayscale the unprocessed image
    gray_img = rgb2gray(unprocessed_img)
    gray_img = np.expand_dims(gray_img, 2)

    # Convert the gray_img into a three channeled image again
    gray_img = np.concatenate((gray_img, gray_img, gray_img), 2)
    gray_img = np.uint8(gray_img)

    # Make the gray image even more faint (Fade the image even more)
    if vis != 'cam':
        for i in range(len(gray_img)):
            for j in range(len(gray_img[0])):
                if gray_img[i][j][0] < 235:
                    gray_img[i][j][0] = 235
                    gray_img[i][j][1] = 235
                    gray_img[i][j][2] = 235

    # Lets overlay the heatmap onto original image. CAM has its own colormap,
    # All other techniques use a modified red colormap
    if vis == 'cam':
        colored_heatmap = np.uint8(matplotlib.cm.jet(normalized)[..., :3] * 255)
    else:
        # Custom colormap to increase the visibility of the gradients

        reds = matplotlib.cm.get_cmap('Reds', 256)
        newcolors = reds(np.linspace(0, 1, 256))
        newcolors[0, :] = 1
        newcolors = newcolors ** contrast

        newcmp = matplotlib.colors.ListedColormap(newcolors)

        colored_heatmap = np.uint8(newcmp(normalized)[..., :3] * 255)

    # Cam even has its own overlay technique.
    if vis == 'cam':
        overlaid = overlay(colored_heatmap, gray_img, alpha=0.3)
    else:
        overlaid = np.minimum(colored_heatmap, gray_img)

    # First Subplot
    ax[0].set_title("Neuron {} activation: %.6f".format(neuron) % pred[neuron])
    ax[0].imshow(unprocessed_img)

    # Second subplot
    ax[1].set_title("Importance map for neuron {}:".format(neuron))
    if vis == 'cam':
        heatmap = ax[1].imshow(visualization, cmap='jet')
    else:
        heatmap = ax[1].imshow(visualization, cmap=newcmp)
    f.colorbar(heatmap, ax=ax[1])

    # Third subplot
    ax[2].set_title("Overlaid Image")
    ax[2].imshow(overlaid)
    f.colorbar(heatmap, ax=ax[2])


# TODO: Is name necessary??
def visualize(model, img, unprocessed_img, vis, name, conv_layer=None,
              background=None, show=False, title=None, clip=2, contrast=2,
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
    :param clip: (int) Used with "shap" or "integrated_grad". Clip larger gradients
    to max/clip, so smaller gradients are also displayed.
    Should be between 1-10.
    :param contrast: (float) Used with "shap" or "integrated_grad". Determines
    how much should smaller gradients show. (Higher contrast = smaller gradients
    show more.)
    :param neuron: (int) Which neuron to display. If unset, will display the
    neuron with highest activation.
    :return:
    '''

    # Dimension
    dim = len(model.input.shape) - 1

    # Make predictions (will be used later)
    pred = model.predict(np.array([img]))[0]
    if neuron is None:
        neuron = np.argmax(pred)

    # Swap softmax with linear. Makes gradients more visible
    model.layers[-1].activation = activations.linear
    model = utils.apply_modifications(model)

    # For each output option, visualize which inputs effect it.
    if vis == 'cam':
        if conv_layer is None:
            print("Warning! If your model is Resnet or has a global average pooling"
                  "layer, conv_layer shouldn't be None or else CAM won't work!")
        # Get visualization of last layer
        layer_idx = -1

        visualization = visualize_cam(model, layer_idx, filter_indices=neuron,
                              seed_input=img, penultimate_layer_idx=conv_layer,
                              backprop_modifier=None)
    elif vis == 'saliency':
        # Get visualization of last layer
        layer_idx = -1

        visualization = visualize_saliency(model, layer_idx,
                                           backprop_modifier='guided',
                                           filter_indices=neuron,
                                           seed_input=img)
    elif vis == 'shap':
        assert background is not None, "Shap requires a background image."

        visualization = visualize_shap(model, img, background,
                                       clip, neuron)

    elif vis == 'integrated_grad':
        assert background is not None, "Integrated Gradients requires a background"

        visualization = visualize_integrated_gradients(model, img, background,
                                                       clip, neuron)

    if dim == 2:
        display_2d(visualization, img, neuron, pred, title)
    elif dim == 3:
        display_3d(visualization, unprocessed_img, neuron, pred, title, vis, contrast)
    else:
        raise Exception("Cannot display a model which isn't 2D or 3D!")

    print("Saving Figure to {}_explained_{}.png".format(name, vis))
    plt.savefig('{}_explained_{}.png'.format(name, vis))

    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, help='path of keras model')
    parser.add_argument('--unprocessed_img', type=str, help='path of raw image.')
    parser.add_argument('--preprocessed_img', type=str,
                        help='path of image. Must be npy file and preprocessed.')
    parser.add_argument('--vis', type=str, default='cam',
                        help='either cam, shap, integrated_grad, or saliency')
    parser.add_argument('--conv_layer', type=int, default=None,
                        help='Used for CAM. Index of last convolutional layer')
    parser.add_argument('--background', type=str,
                        help='path of "background" image')
    parser.add_argument('--show', type=bool, default=False,
                        help='whether to display the image')
    parser.add_argument('--clip', type=int, default=2,
                        help='Used for shap or integrated_grad. '
                             'Sensitvity of display.')
    parser.add_argument('--contrast', type=float, default=2,
                        help='Used for shap or integrated_grad. '
                             'How much smaller gradients are emphasized.')
    parser.add_argument('--neuron', type=int, default=None,
                        help='Which neuron to visualize. If blank, will '
                             'visualize the neuron with highest activation')
    # TODO: Make this a boolean once you figure out booleans
    # TODO: There seems to be a bug with show always showing.

    args = parser.parse_args()

    assert args.vis in ['cam', 'saliency', 'shap', 'integrated_grad'], \
        "vis must be cam, shap, integrated_grad or saliency!"

    # Load the model
    model = load_model(args.model)

    dimension = len(model.input.shape) - 1

    # Load the image

    if dimension == 2:
        preprocessed_img = np.load(args.preprocessed_img)
        unprocessed_img = np.load(args.unprocessed_img)
    else:
        preprocessed_img = np.load(args.preprocessed_img)
        unprocessed_img = utils.load_img(args.unprocessed_img)

    # Load the background if there is one
    # Note: Background must be preprocessed!
    if args.background:
        background = np.load(args.background)
    else:
        background = None

    visualize(model, preprocessed_img, unprocessed_img, args.vis,
              args.unprocessed_img[:-4], conv_layer=args.conv_layer,
              background=background, show=args.show, clip=args.clip,
              contrast=args.contrast, neuron=args.neuron)
