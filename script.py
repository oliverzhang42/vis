# Things to do:
#
# 1. Run a different file
# 2. Collect results and store them somewhere
# 3. Run automatic tests on the different models
# 4. Store these results
# 5. Wipe everything clean, pick a different model and start again.

import os
import shutil
import pudb

folder = "cheng_models"

files = []
# r=root, d=directories, f=files
for r, d, f in os.walk(folder):
    for file in f:
        if '.hdf5' in file:
            files.append(os.path.join(r, file))

f = open("results.txt", "w")

import pudb; pudb.set_trace()

for m_path in files:
    if m_path[-12:-10] != '10':
        continue
    
    os.system("python3 validation_over_ppg.py --vis saliency --model {} --ref images_1d/half.npy --dim 1".format(m_path))
    m_name = m_path.split("/")[-1]
    m_name = m_name[:-5] + ".npz"
    os.system("mv saliency/history.npz histories/{}".format(m_name))

    f.write("{} ".format(m_path))

    # Congruence
    cong_str = os.popen("python3 congruence.py --path histories/{} --abs True".format(m_name)).read()
    cong = float(cong_str.split(" ")[-1])

    f.write("congruence: {} ".format(cong))

    # Validation Metrics
    str_ = "python3 validation_metrics.py --path histories/{}".format(m_name) + " --abs True --val {}"

    val = ['sectional', 'interval', 'pixel']
    
    for metric in val:
        x = str_.format(metric)
        val_str = os.popen(x).read()
        val_metric = float(val_str.split(" ")[-1])
        
        
        f.write("{}: {} ".format(metric, val_metric))

    f.write("\n")
    f.flush()
