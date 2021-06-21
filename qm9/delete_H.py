import numpy as np
import os

rename = False
if rename:
    path = './temp/qm9'
    files = os.listdir(path)
    for file in files:
        if os.path.isfile(os.path.join(path,file)) and file[-4:] == '.npz':
            old_name = os.path.join(path, file)
            new_name = os.path.join(path, file[:-4] + '_org' + file[-4:])
            os.rename(old_name, new_name)

