import json
import matplotlib as plt
from multiprocessing import Process, Queue
import numpy as np
import os
import pandas as pd
from vis.utils import utils
from visualize import *


def parse_timestamps(timestamps):
    '''
    Parses the human annotations from the timestamps

    inputs:
    timestamps (list): Formatted like in "Signal Quality Index.csv"

    outputs:
    human_annotations (list): An array of integers between 0 and 7200,
                              the indices of the human annotations.
    '''
    human_annotations = []

    for timestamp in timestamps:
        array = []
        timestamp = json.loads(timestamp)

        for d in timestamp:
            array.append(int(float(d['start']) * 7201 / 30))
            array.append(int(float(d['end']) * 7201 / 30))

        array.sort()
        human_annotations.append(array)

    return np.array(human_annotations)


def parse_csv(path):
    '''
    Helper function to parse the csv files in 1D_data_and_model/CSVFiles.
    '''
    f = open(path)
    data = f.read()[:-1].split(',')
    data = np.array(data).astype('float32')
    data = np.reshape(data, (7201, 1))
    return data


def read_noisy_data(data_names):
    '''
    Function which inputs the names of the data (eg: P04-203) and attempts to
    see if it exists in the 1D_data_and_model/CSVFiles folder. If it not only exists
    but is also bad data, (labeled with 'B_'), then we return it.
    
    (For example, checks if B_A_P04-203 or B_N_P04-203 exists)
    
    inputs:
    data_names (list of str): A list of the names to check.

    outputs:
    noisy_data (np.array): the numpy array of the noisy data.
    indices (list): list of the indices of the noisy data.
    '''

    headings = ['B_A_', 'B_N_']
    noisy_data = []
    indices = []

    for i in range(len(data_names)):
        name = data_names[i]
        data = None

        for heading in headings:
            full_name = heading + name
            data_path = os.path.join("1D_data_and_model/CSVfiles", full_name) + '.csv'

            if os.path.exists(data_path):
                data = parse_csv(data_path)
                noisy_data.append(data)
                indices.append(i)
                break

        if data is None:
            print("Image: {} does not exist!".format(name))

    return np.array(noisy_data), indices


def visualize_over_dataset(vis_type, model_path, dataset, neuron, ref=None):
    '''
    Make local visualizations over the whole dataset. Only works with 1d datapoints.
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
    background = np.load(ref)

    for i in range(len(dataset)):
        data = dataset[i]
        queue = Queue()
        p = Process(target=visualize_wrapper, args=(queue, data, background, neuron, vis_type))
        p.start()
        p.join()
        p.terminate()
        del p
        visualization = queue.get()

        print("Processed datapoint number {}".format(i))

        vis_history.append(visualization)

    return np.array(vis_history)


def visualize_wrapper(queue, data, background, neuron, vis_type):
    '''
    Helper function for visualize_over_dataset. Because its only the child process
    which calls tensorflow and not the parent process, the memory of tensorflow is
    released every time. Unfortunately, it also means we have to load the model
    every time, but oh well.
    '''

    import keras
    m = keras.models.load_model("resnet_ppg_1d")

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
    '''
    vis_type: out of "saliency", "shap", or "integrated_gradients" what type
              of visualziation would you like?
    model_path: path to model
    background: if using "shap" or "integrated_gradients" you need a reference
                image. Set background to the path of the reference image.
    neuron: index of neuron in the final layer of model to visualize
    save_path: where to save everything once you're done.
    '''
    vis_type = "saliency"
    model_path = "resnet_ppg_1d"
    background = "images_1d/half.npy"
    neuron = 0

    print("Visualizing model {} using {} technique!".format(model_path, vis_type))

    df = pd.read_csv("Signal Quality Index.csv")

    paths = df.FilePath.tolist()
    timestamps = df.Value.tolist()
 
    names = []
    for path in paths:
        names.append(path[4:-7])
    
    # The human annotations, formatted as a list of indices.
    human_annotations = parse_timestamps(timestamps)
    noisy_dataset, noise_indices = read_noisy_data(names)
    human_annotations = human_annotations[noise_indices]

    vis_history = visualize_over_dataset(vis_type, model_path, noisy_dataset, neuron, background)
    
    if not os.path.isdir(vis_type):
        print("Creating Directory {}".format(vis_type))
        os.mkdir(vis_type)

    save_path = os.path.join(vis_type, "history")
    np.savez(save_path, vis_history, noisy_dataset, human_annotations)
