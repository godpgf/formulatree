import os
from formulatree import *

feature_path = "feature"
out_feature_path = "output"
model_path = "model"

if __name__ == '__main__':
    if not os.path.exists(out_feature_path):
        os.mkdir(out_feature_path)
    files = os.listdir(feature_path)
    for file in files:
        if not os.path.isdir(file) and file.endswith(".txt"):
            from_feature_path = feature_path + "/" + file
            to_feature_path = out_feature_path + "/" + file
            def pred_fun(line):
                with open(to_feature_path, 'a') as f:
                    f.write("%s\n" % line)
            cur_model_path = model_path + "/" + file[:-4]
            pred_alphatree_list(from_feature_path, pred_fun, model_path=cur_model_path + "/model.ckpt", epoch_num=60000)

