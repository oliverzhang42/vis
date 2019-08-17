import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from visualize import display_1d
from validation_metrics import preprocess

mpl.rcParams['figure.figsize'] = [12.0, 8.0]

'''
Script with displays the results of history.npz.

Currently uses the absolute value preprocessing.
'''


f = np.load("history.npz", allow_pickle=True)
vis_history = f.get("arr_0")
vis_history = preprocess(vis_history, abs_=True)
img_history = f.get("arr_1")
vbar_history = f.get("arr_2")

for i in range(3):
    index = np.random.choice(range(len(vis_history)))

    display_1d(vis_history[index], img_history[index], 0, [0,1], vbar_history[index], vertical=vbar_history[index])
    plt.show()

