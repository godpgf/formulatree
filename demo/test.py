#coding=utf-8
#author=godpgf
import re
from formulatree import *

if __name__ == '__main__':
    with open("doc/alpha101.txt", 'r') as f:
        line = f.readline()
        tree_list = []
        while line:
            line = re.search(r"(?P<alpha>\w+): (?P<content>.*)", line).group('content')
            tree = FTree.decode(line)
            tree_list.append(tree)
            new_line = FTree.encode(tree)
            if line != new_line:
                print(line)
                print(new_line)

            line = f.readline()
        vocab = FTree.get_vocab(tree_list)
        ml_tree_list = []
        for tree in tree_list:
            ml_tree_list.extend(tree.get_ml_tree_list(vocab))

        train(1000, ml_tree_list, vocab)
        sess, model = load_model(vocab)
        for i in range(100):
            ftree_list = create_formula_tree(sess, model, vocab)
            for ftree in ftree_list:
                print(FTree.encode(ftree))