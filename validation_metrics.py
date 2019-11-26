import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from visualize import display_1d, display_2d

mpl.rcParams['figure.figsize'] = [6.0, 6.0]


def get_null(vis_history):
    '''
    Returns index array of all visualizations whose only value is zero.
    '''
    indices = []

    max_ = vis_history.max(1)
    min_ = vis_history.min(1)

    for i in range(len(vis_history)):
        if max_[i] == 0 and min_[i] == 0:
            indices.append(i)

    return indices


def preprocess(vis_history, abs_=False):
    '''
    Preprocesses vis_history. If doing absolute preprocessing then takes
    the absolute values and normalizes them to between 0 and 1 by dividing
    by maximum.

    If doing regular preprocessing them to between 0 and 1 such that what
    once was a 0 is now a 0.5. In detail, the procedure is to scale the data
    around 0 to between -0.5 and 0.5. Then we add 0.5
    '''

    if abs_:
        preprocessed = np.abs(vis_history)
        preprocessed = preprocessed - preprocessed.min(1)[:, np.newaxis]
        preprocessed = preprocessed / preprocessed.max(1)[:, np.newaxis]
        return preprocessed
    else:
        vis_history_max = vis_history.max(1)
        vis_history_min = vis_history.min(1)
        normalizer = 2 * np.maximum(vis_history_max, np.abs(vis_history_min))
 
        preprocessed = vis_history / normalizer[:, np.newaxis]
        preprocessed = preprocessed + 0.5
        return preprocessed


def create_roc(pred, test_cases, plot=True):
    '''
    Helper function to create and plot the roc curve.
    '''
    
    fpr, tpr, _ = metrics.roc_curve(test_cases,  pred)
    auc = metrics.roc_auc_score(test_cases, pred)
    if plot:
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.legend(loc=4)
    else:
        print("AuC is: {}".format(str(auc)))

def validate_pixel(vis, annotated):
    '''
    Function which applies pixel wise validation on a single example.

    inputs:
    vis (np.array): the visualization of a single training example.
    annotated (list): the array of human annotations denoting the noise.

    outputs:
    pred (list): the values between 0 to 1 used to make predictions 
                 of which pixels are in the human attention.
    test_cases (list): the true answers to which pixels are in the human
                      attention. 1 means human annotated pixel, 0 means
                      not human annotated pixel.
    '''
    pred = vis.flatten().tolist()
    test_cases = []

    for i in range(annotated[0]):
        test_cases.append(0)

    for i in range((len(annotated) // 2) - 1):
        for j in range(annotated[2*i], annotated[2*i+1]):
            test_cases.append(1)

        for j in range(annotated[2*i+1], annotated[2*i+2]):
            test_cases.append(0)

    for i in range(annotated[-2], annotated[-1]):
        test_cases.append(1)

    for i in range(annotated[-1], len(vis)):
        test_cases.append(0)

    return pred, test_cases


def validate_sectional(vis, annotated):
    '''
    Function which applies sectional validation on a single example.

    input:
    vis (np.array): the visualization of a single training example.
    annotated (list): the array of human annotations denoting the noise.

    outputs:
    pred (list): the values between 0 to 1 used to make predictions
                 of which sections are in the human attention
    test_cases (list): the true answers to which sections are in the human
                       attention. 1 means human annotated section, 0 means
                       not human annotated section.
    '''

    pred = []
    test_cases = []

    if annotated[0] != 0:
        pred.append(np.max(vis[0:annotated[0]]))
        test_cases.append(0)

    for i in range((len(annotated) // 2) - 1):
        start_noisy = annotated[2*i]
        end_noisy = annotated[2*i+1]
        start_clean = end_noisy
        end_clean = annotated[2*i+2]

        if start_noisy != end_noisy:
            pred.append(np.max(vis[start_noisy:end_noisy]))
            test_cases.append(1)

        if start_clean != end_clean:
            pred.append(np.max(vis[start_clean:end_clean]))
            test_cases.append(0)
    
    if annotated[-2] != annotated[-1]:
        pred.append(np.max(vis[annotated[-2]:annotated[-1]]))
        test_cases.append(1)
    
    if not annotated[-1] >= len(vis):
        pred.append(np.max(vis[annotated[-1]:]))
        test_cases.append(0)

    return pred, test_cases


def validate_interval(vis, annotated):
    '''
    Function which applies interval validation on a single example.
    Note: Currently the number of intervals is hardcoded.

    inputs:
    vis (np.array): the visualization of a single training example.
    annotated (list): the array of human annotations denoting the noise.

    outputs:
    pred (list): the values between 0 to 1 used to make predictions
                 of which sections are in the human attention
    test_cases (list): the true answers to which intervals are in the human
                       attention. 1 means human annotated interval, 0 means
                       not human annotated interval.
    '''

    intervals = 6
    interval_size = len(vis) // intervals

    pred = []
    test_cases = []
    bad = []

    for j in range(len(annotated) // 2):
        start = annotated[2*j]
        end = annotated[2*j+1]

        for k in range(start, end):
            bad.append(k)

    for j in range(intervals):
        start = interval_size*j
        end = interval_size*(j+1)

        pred.append(np.max(vis[start:end]))

        # 0 denotes a good section, 1 denotes a noisy section
        section_quality = 0

        for k in range(start, end):
            if k in bad:
                section_quality = 1
                break

        test_cases.append(section_quality)

    return pred, test_cases


def validate_dataset(vis_history, annotations, val_type):
    '''
    Calculates pixel, sectional, or interval AuROC over a whole dataset.

    inputs:
    vis_histroy (list): A list with all the visualizations over a dataset.
    annotations (list): A list with the human annotations over the dataset.
    val_type (str): Which type of AuROC validation do you want
    '''

    assert val_type in ['pixel', 'sectional', 'interval']
    
    pred = []
    test_cases = []

    for i in range(len(vis_history)):
        vis_example = vis_history[i]
        annotated = annotations[i]
        annotated.sort()
        
        if val_type == 'pixel':
            pred_example, test_case = validate_pixel(vis_example, annotated)
        elif val_type == 'sectional':
            pred_example, test_case = validate_sectional(vis_example, annotated)
        elif val_type == 'interval':
            pred_example, test_case = validate_interval(vis_example, annotated)
        else:
            raise Exception("val_type must be 'pixel', 'sectional', or 'interval!'")

        # Note: this is concatenation.
        pred = pred + pred_example
        test_cases = test_cases + test_case

    return pred, test_cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments.')
    parser.add_argument('--path', type=str, required=True,
                        help='path of history npz.')
    parser.add_argument('--abs', type=bool, default=False,
                        help='whether to use the absolute preprocessing.')
    parser.add_argument('--val', type=str, default='sectional', 
                        help='Validation type. Can be sectional, pixel, or interval.')
    parser.add_argument('--show', type=bool, default=False,
                        help='Whether to show the ROC curve or just print the AuC.')

    args = parser.parse_args()

    history = np.load(args.path, allow_pickle=True)

    vis_history = history.get("arr_0")
    img_history = history.get("arr_1").astype('float32')
    annotations = history.get("arr_2")

    # If dimension is 2, we reduce the 224 x 224 visualization 
    # to a 224 one by taking the maximum over the columns.
    if len(vis_history.shape) == 3:
        print("Detected 2d images, taking max over columns!")
        vis_history = vis_history.max(1)
    
    null_indices = get_null(vis_history)

    if null_indices != []:
        print("Removing {} examples for being all zero".format(len(null_indices)))
        indices = []

        for i in range(len(vis_history)):
            if i not in null_indices:
                indices.append(i)

        vis_history = vis_history[indices]
        img_history = img_history[indices]
        annotations = annotations[indices]

    vis_history = preprocess(vis_history, abs_=args.abs)
    pred, test_cases = validate_dataset(vis_history, annotations, args.val)

    create_roc(pred, test_cases, plot=args.show)
    if args.show:
        plt.show()
        
