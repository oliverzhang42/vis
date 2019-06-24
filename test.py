from keras.applications import ResNet50
from keras import activations
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import numpy as np
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam
plt.rcParams['figure.figsize'] = (15, 10)

# Build the ResNet50 network with ImageNet weights
model = ResNet50(weights='imagenet', include_top=True)

# Swap softmax with linear
model.layers[-1].activation = activations.linear
model = utils.apply_modifications(model)

img1 = utils.load_img('ouzel1.jpg', target_size=(224, 224))
img2 = utils.load_img('ouzel2.jpg', target_size=(224, 224))

#f, ax = plt.subplots(2, 2)
#ax[0, 0].imshow(img1)
#ax[0, 1].imshow(img2)

# Visualize the last layer
layer_idx = -1

'''
for i, img in enumerate([img1, img2]):
    # 20 is the imagenet index corresponding to `ouzel`
    grads = visualize_saliency(model, layer_idx, filter_indices=20, seed_input=img)

    # visualize grads as heatmap
    ax[1, i].imshow(grads, cmap='jet')

#'''
'''
for modifier in ['guided', 'relu']:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle(modifier)
    for i, img in enumerate([img1, img2]):
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_saliency(model, layer_idx, filter_indices=20,
                                   seed_input=img, backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        ax[i].imshow(grads, cmap='jet')
#'''
# =========================================

''' (Alright, something's not working, check later)
penultimate_layer = None #utils.find_layer_idx(model, 'res5c_branch2c')

for modifier in [None, 'guided', 'relu']:
    plt.figure()
    f, ax = plt.subplots(1, 2)
    plt.suptitle("vanilla" if modifier is None else modifier)
    for i, img in enumerate([img1, img2]):
        # 20 is the imagenet index corresponding to `ouzel`
        grads = visualize_cam(model, layer_idx, filter_indices=20,
                              seed_input=img, penultimate_layer_idx=penultimate_layer,
                              backprop_modifier=modifier)
        # Lets overlay the heatmap onto original image.
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        ax[i].imshow(overlay(jet_heatmap, img))
#'''

plt.show()