#coding=utf-8
#author=godpgf

from .ftree import FTree, FNode
from .ml import MLNode, MLTree, create_model, train_model, load_model, save_model
import random
import numpy as np
import tensorflow as tf
import gc

def create_formula_tree(sess, model, vocab, batch_size = 128, max_node_size = 40):
    root_list = [FNode('none') for i in range(batch_size)]
    ml_tree_cache_list = [None for i in range(batch_size)]
    # 当前没有完成的节点
    all_node_list = [[root_list[i]] for i in range(batch_size)]
    cur_node_id_list = [[0] for i in range(batch_size)]
    for i in range(max_node_size):
        for j in range(batch_size):
            if len(cur_node_id_list[j]) > 0:
                ml_tree_cache_list[j] = FTree.parse_ml_tree_from_leaf(all_node_list[j], cur_node_id_list[j][-1])
                MLNode.mend_ml_tree(ml_tree_cache_list[j], vocab)
                ml_tree_cache_list[j] = MLTree(ml_tree_cache_list[j])

        # 得到批次是0的预测，其实批次也就只有0 (^_^)
        pred_y_list = model.evaluate(ml_tree_cache_list, sess)[0]

        for j in range(batch_size):
            if len(cur_node_id_list[j]) > 0:
                #创建新节点
                pred_y = pred_y_list[j]
                node_list = all_node_list[j]
                _choose_node(node_list, cur_node_id_list[j], vocab, pred_y)

    # 漏斗筛选出没有超长的树
    ftree_list = []
    for j in range(batch_size):
        if len(cur_node_id_list[j]) == 0:
            node_list = all_node_list[j]
            node_list.reverse()
            last_node_list = []
            reverse_id = lambda id, l: -1 if id == 0 else (l - id - 1)
            #去掉第一个虚拟节点，然后反转
            for i in range(len(node_list) - 1):
                node = node_list[i]
                last_node_list.append(node)
                node.pre_id = reverse_id(node.pre_id, len(node_list))
                for j in range(len(node.children)):
                    node.children[j] = reverse_id(node.children[j], len(node_list))
            ftree_list.append(FTree(last_node_list))
    return ftree_list





def _choose_node(node_list, cur_node_id, vocab, pred_y):
    all_value = random.random() * (np.sum(pred_y) - pred_y[0])
    arg_index = 1
    assert all_value >= 0.0 and arg_index < len(pred_y)
    while all_value >= 0.0 and arg_index < len(pred_y):
        all_value -= pred_y[arg_index]
        arg_index += 1
    arg_index -= 1
    opt = vocab.idx2word[arg_index]
    coff = None if opt not in vocab.coff_dict else random.choice(vocab.coff_dict[opt])

    #同一个操作不应该同时出现两次
    cur_node = node_list[cur_node_id[-1]]
    child_num = vocab.node_dict[opt]
    if child_num == 1 and cur_node.name == opt:
        if arg_index > 1:
            arg_index -= 1
        else:
            arg_index += 1
        opt = vocab.idx2word[arg_index]
        coff = None if opt not in vocab.coff_dict else random.choice(vocab.coff_dict[opt])

    #插入节点
    new_node = FNode(opt, coff)
    new_node_id = len(node_list)
    new_node.pre_id = cur_node_id[-1]
    new_node.pre_child_index = len(cur_node.children)
    cur_node.add_child(new_node_id)
    node_list.append(new_node)

    #删除当前节点，它不需要再
    if cur_node.name == 'none' or len(cur_node.children) == vocab.node_dict[cur_node.name]:
        #孩子已经满了
        cur_node_id.pop()

    #加入新的待分析节点
    if vocab.node_dict[opt] > 0:
        cur_node_id.append(new_node_id)


def train_alphatree_list(path, model_path="save/model.ckpt", train_epoch = 10000, max_node_size = 24):
    alphatree_list = []
    with open(path, 'r') as r:
        line = r.readline()
        while line:
            if len(line) > 2:
                alphatree_list.append(line[:-1])
            line = r.readline()
    tree_list = [FTree.decode(factor) for factor in alphatree_list]
    vocab = FTree.get_vocab(tree_list)
    ml_tree_list = []
    for tree in tree_list:
        ml_tree_list.extend(tree.get_ml_tree_list(vocab))

    with tf.Graph().as_default():
        model = create_model(vocab, len(ml_tree_list), max_node_size)
        with tf.Session() as sess:
            train_model(sess, train_epoch, ml_tree_list, model)
            save_model(sess, model_path)


def pred_alphatree_list(path, pred_fun, model_path="save/model.ckpt", epoch_num = 10000, max_node_size = 24):
    alphatree_list = []
    with open(path, 'r') as r:
        line = r.readline()
        while line:
            if len(line) > 2:
                alphatree_list.append(line[:-1])
            line = r.readline()
    tree_list = [FTree.decode(factor) for factor in alphatree_list]
    vocab = FTree.get_vocab(tree_list)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            model = load_model(sess, vocab, model_path, max_node_size=max_node_size)

            for _ in range(epoch_num):
                if _ % 100 == 0:
                    print('gc')
                    gc.collect()
                ftree_list = create_formula_tree(sess, model, vocab, max_node_size=max_node_size)
                for ftree in ftree_list:
                    line = FTree.encode(ftree)
                    pred_fun(line)