import numpy as np
from validation_metrics import preprocess

def congruence(vis_history, annotations):
    '''
    Calculates the congruence of the human
    annotations and the model annotations.

    inputs:
    vis_history (list): The total model attention.
    annotations (list): The human annotations.

    outputs:
    avg_congruence (int): The average congruence per image
                      between human and model annotations.
    '''

    cong_per_img = []

    for i in range(len(vis_history)):
        vis = vis_history[i]
        anno = annotations[i]
        
        correct_attention = 0
        total_attention = np.sum(vis)

        for j in range(len(anno)//2):
            start = anno[2*j]
            end = anno[2*j+1]

            correct_attention += np.sum(vis[start:end])

        cong_per_img.append(correct_attention / total_attention)

    return np.mean(cong_per_img)


if __name__ == '__main__':
    history_path = 'shap/history.npz'
    absolute_values = False
    history = np.load(history_path, allow_pickle=True)

    vis_history = history.get("arr_0")
    annotations = history.get("arr_2")

    vis_history = preprocess(vis_history, abs_=absolute_values)
    cong = congruence(vis_history, annotations)

    print("Congruence is: {}".format(cong))
