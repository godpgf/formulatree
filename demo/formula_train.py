import os
from formulatree import *

feature_path = "feature"
model_path = "model"

if __name__ == '__main__':
    files = os.listdir(feature_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    for file in files:
        if not os.path.isdir(file) and file.endswith(".txt"):
            path = feature_path + "/" + file
            cur_model_path = model_path + "/" + file[:-4]
            if not os.path.exists(cur_model_path):
                os.mkdir(cur_model_path)
            train_alphatree_list(path, model_path=cur_model_path + "/model.ckpt")