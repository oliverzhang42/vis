import argparse
import json
import matplotlib
from multiprocessing import Process, Queue
import numpy as np
import os
import pandas as pd
from vis.utils import utils
from visualize import *


def parse_timestamps(timestamps, datapoints):
    '''
    Parses the human annotations from the timestamps.
    Note that this is hardcoded to deal with 30 second PPG signals.

    inputs:
    timestamps (list): Formatted like in "Signal Quality Index.csv"
    datapoints (int): The number of datapoints per example. For instance
                      with our 240 Hz sampling rate, a 30 second PPG signal
                      becomes 7201 datapoints. But when using the 2d model,
                      the effective number of datapoints is 224 because we
                      simplify the images into 224x224x3 images.

    outputs:
    human_annotations (list): An array of integers between 0 and 7200,
                              the indices of the human annotations.
    '''
    human_annotations = []

    for timestamp in timestamps:
        array = []
        timestamp = json.loads(timestamp)

        for d in timestamp:
            array.append(int(float(d['start']) * datapoints / 30))
            array.append(int(float(d['end']) * datapoints / 30))

        array.sort()
        human_annotations.append(array)

    return np.array(human_annotations)


def preprocess_img(img, mean_image_path):
    '''
    Preprocesses image, specific to the 2d resnet model.
    Subtracts the mean image, then divides by 128.
    '''
    img = img - np.load(mean_image_path)
    img = img / 128.0
    return img


def parse_csv(path):
    '''
    Helper function to parse the csv files in 1D_data_and_model/CSVFiles.
    '''
    f = open(path)
    data = f.read()[:-1].split(',')
    data = np.array(data).astype('float32')
    data = np.reshape(data, (7201, 1))
    return data


def read_noisy_data(data_names, folder_name, dimension, mean_image_path=None):
    '''
    Function which inputs the names of the data (eg: P04-203) and attempts to
    see if it exists in the folder_name folder. If it not only exists
    but is also bad data, (labeled with 'B_'), then we return it.
    
    (For example, checks if B_A_P04-203 or B_N_P04-203 exists)
    
    inputs:
    data_names (list of str): A list of the names to check.
    folder_name (str): The path to the folder with all the data.
    dimension (int): The dimension of the convolutional model
    mean_image_path (str): Optional argument only used if using a 2d
                           model. The mean image for preprocessing.

    outputs:
    noisy_data (np.array): the numpy array of the noisy data.
    indices (list): list of the indices of the noisy data.
    '''

    headings = ['B_A_', 'B_N_']
    noisy_data = []
    indices = []

    if dimension == 1:
        ending = '.csv'
    elif dimension == 2:
        ending = '.jpg'
    else:
        raise Exception("I don't know how to handle non 1d or 2d models!")

    for i in range(len(data_names)):
        name = data_names[i]
        data = None

        for heading in headings:
            full_name = heading + name
            data_path = os.path.join(folder_name, full_name) + ending

            if os.path.exists(data_path):
                if dimension == 1:
                    data = parse_csv(data_path)
                elif dimension == 2:
                    data = matplotlib.image.imread(data_path)
                    data = preprocess_img(data, mean_image_path)
                else:
                    raise Exception("I don't know how to handle non 1d or 2d models!")

                noisy_data.append(data)
                indices.append(i)
                break

        if data is None:
            print("Image: {} does not exist!".format(name))

    return np.array(noisy_data), indices


def visualize_over_dataset(vis_type, model_path, dataset, neuron, ref=None):
    '''
    Make local visualizations over the whole dataset.
    Note: The program opens a new process everytime it calls tensorflow. Otherwise
    there is a memory leak. 

    inputs:
    vis_type (str): 'saliency', 'integrated_gradients', or 'shap'. Which type of vis
                     to use.
    model_path (str): path to model. Again, must be 1d model.
    dataset (np.array): dataset to visualize. If need preprocessing, it must be done
                        outside of this function.
    neuron (int): Which neuron in the final layer to visualize
    ref (np.array): reference image if using shap or integrated gradients

    outputs:
    vis_history (np.array): visualizations over all the dataset.
    '''

    assert vis_type in ['saliency', 'integrated_gradients', 'shap'], "vis_type is wrong!"

    vis_history = []

    if not ref is None:
        background = np.load(ref)
    else:
        background = None

    for i in range(len(dataset)):
        data = dataset[i]
        queue = Queue()
        p = Process(target=visualize_wrapper, args=(queue, model_path, data, background, neuron, vis_type))
        p.start()
        visualization = queue.get()
        p.join()
        del p

        print("Processed datapoint number {}".format(i))

        vis_history.append(visualization)

    return np.array(vis_history)


def visualize_wrapper(queue, model_path, data, background, neuron, vis_type):
    '''
    Helper function for visualize_over_dataset. Because its only the child process
    which calls tensorflow and not the parent process, the memory of tensorflow is
    released every time. Unfortunately, it also means we have to load the model
    every time, but oh well.
    '''

    import keras
    m = keras.models.load_model(model_path)

    if vis_type == 'saliency':    
        vis = visualize_saliency(m, -1, backprop_modifier='guided', 
                                 filter_indices=neuron, seed_input=data)
    elif vis_type == 'shap':
        vis = visualize_shap(m, data, background, neuron)
    elif vis_type == 'integrated_gradients':
        vis = visualize_integrated_gradients(m, data, background, neuron)
    else:
        raise Exception("vis_type needs to be saliency, shap, or integrated_gradients")

    queue.put(vis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--vis', type=str, default='shap', 
        help='Out of "saliency", "shap", or "integrated_gradients the type of visualization.')
    parser.add_argument('--model', type=str, required=True, help='path to model')
    parser.add_argument('--ref', type=str,
        help='if using "shap" or "integrated_gradients you need a reference image. This is its path.')
    parser.add_argument('--neuron', type=int, default=0,
        help='Index of neuron in the final layer of model to visualize.')
    parser.add_argument('--dim', type=int, required=True,
        help='Dimension of model, only handles 1 or 2 right now.')
    parser.add_argument('--mean_img', type=str, 
        help='path of the mean_image for 2d preprocessing.')

    args = parser.parse_args()
    '''
    Arguments determined by dimension:

    datapoints: length of a single training example. 7201 for 1d models,
                224 for 2d models.
    folder: folder with all the data. pure_test/CSVfiles for 1d, pure_test/pure_test_plot
            for 2d. For 1d it expects csv files, for 2d it expects jpg images.
    save_folder: folder to save all visualizations in.
    '''
    assert args.vis in ['saliency', 'integrated_gradients', 'shap'], \
            "args.vis must be either saliency, integrated_gradients, or shap!"

    if args.dim == 1:
        datapoints = 7201
        folder = "pure_test/CSVfiles"
        save_folder = args.vis
    elif args.dim == 2:
        assert not args.mean_img is None, "If the model is 2d, you need a mean_image \
                for preprocessing!"
        datapoints = 224
        folder = "pure_test/pure_test_plot"
        save_folder = args.vis + '_images'
    else:
        raise Exception("I don't know how to deal with a model not 1d or 2d!")

    print("Visualizing model {} using {} technique!".format(args.model, args.vis))

    df = pd.read_csv("Signal Quality Index.csv")
    paths = df.FilePath.tolist()
    timestamps = df.Value.tolist()
    names = []
    for path in paths:
        names.append(path[4:-7])
    
    # Parsing the csvfiles and dataset 
    # The human annotations, formatted as a list of indices.
    human_annotations = parse_timestamps(timestamps, datapoints)
    noisy_dataset, noise_indices = read_noisy_data(names, folder, args.dim, args.mean_img)
    human_annotations = human_annotations[noise_indices]
    
    vis_history = visualize_over_dataset(args.vis, args.model, noisy_dataset, args.neuron, args.ref)

    if not os.path.isdir(save_folder):
        print("Creating Directory {}".format(save_folder))
        os.mkdir(save_folder)

    save_path = os.path.join(save_folder, "history")
    np.savez(save_path, vis_history, noisy_dataset, human_annotations)
