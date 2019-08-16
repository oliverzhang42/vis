# This is for computing congruence with both positive and negative gradients
# Congruence with only positive gradients is congruence.py

import numpy as np

history = np.load("saliency_history/history.npz", allow_pickle=True)

vis_history = history.get("arr_0")
proportion_right_history = history.get("arr_1")
pred_history = history.get("arr_2")
vbar_history = history.get("arr_3")
img_history = history.get("arr_4")

total_correctness = []
images = len(vis_history)

total_area_marked = 0

for i in range(images):
    item = vis_history[i]
    total_grad = np.sum(np.abs(item))
    correct_grad = 0

    vbar = vbar_history[i]

    former = 0

    for j in range(len(vbar)//2):
        if left != former:
            correct_grad -= np.sum(item[former:left])

        left = vbar[2*j]
        right = vbar[2*j+1]

        left = int(7201 * left / 30)
        right = int(7201 * right / 30)

        correct_grad += np.sum(item[left:right])
        total_area_marked += (right - left)/7201

        former = left

    if right != 7201:
        correct_grad -= np.sum(item[right:])

    total_correctness.append(correct_grad / total_grad)

print(total_correctness / images)
print(total_area_marked / images)
