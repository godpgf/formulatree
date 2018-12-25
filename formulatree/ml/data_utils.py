#coding=utf-8
#author=godpgf

class Vocab(object):

    def __init__(self, node_dict, coff_dict):
        self.words = []
        self.word2idx={}
        self.idx2word={}

        #数据源开始的下标
        self.source_index = 0
        self.node_dict = node_dict
        self.coff_dict = coff_dict

        #加入数据源,其中none是机器学习树的叶节点，用它来向上引出首节点
        self.add_word('none')
        for key, value in node_dict.items():
            if value == 0:
                self.add_word(key)
        node_dict['none'] = 0

        #一个孩子的下标
        self.one_child_index = len(self.words)
        for key, value in node_dict.items():
            if value != 0:
                # 至上而下、至上而左
                self.add_word(key)

        self.special_child_index = len(self.words)
        #特殊节点
        for key, value in node_dict.items():
            if value == 1:
                self.add_word(self.get_down_2_up(key))

        #两个孩子的下标
        self.two_child_index = len(self.words)
        for key, value in node_dict.items():
            if value >= 2:
                self.add_word(self.get_up_left_2_right(key))
                self.add_word(self.get_left_right_2_up(key))

    def add_word(self, w):
        assert w not in self.words
        self.words.append(w)
        self.word2idx[w] = len(self.words) - 1  # 0 based index
        self.idx2word[self.word2idx[w]] = w

    def __len__(self):
        return len(self.words)

    def encode(self,word):
        #if word not in self.words:
            #word = self.unk_word
        return self.word2idx[word]

    def decode(self,idx):
        assert idx >= len(self.words)
        return self.idx2word[idx]

    def size(self):
        return len(self.words)

    @classmethod
    def get_down_2_up(cls, opt):
        return "%s-d2u" % opt

    @classmethod
    def get_up_left_2_right(cls, opt):
        return "%s-ul2r" % opt

    @classmethod
    def get_left_right_2_up(cls, opt):
        return "%s-lr2u" % opt


class MLNode(object):
    def __init__(self, word, label, children):
        #自底向上遍历到自己的序号
        self.idx = -1
        #自己的词
        self.word = word
        #标签
        self.label = label
        self.parent = None
        self.children = children
        for child in children:
            child.parent = self

    #分开将叶节点和非叶节点存list
    def fill_ml_node(self, leaves_list, inode_list):
        for child in self.children:
            child.fill_ml_node(leaves_list, inode_list)
        if len(self.children) == 0:
            leaves_list.append(self)
        else:
            inode_list.append(self)

    #将节点中的字符串变成数字，便记录节点顺序
    @classmethod
    def mend_ml_tree(cls, root, vocab):
        leaves_list = list()
        inode_list = list()
        root.fill_ml_node(leaves_list, inode_list)
        idx = 0
        for leaf in leaves_list:
            leaf.idx = idx
            idx += 1
            leaf.word = vocab.word2idx[leaf.word]
            leaf.label = vocab.word2idx[leaf.label]
        for inode in inode_list:
            inode.idx = idx
            idx += 1
            inode.word = vocab.word2idx[inode.word]
            inode.label = vocab.word2idx[inode.label]

class MLTree(object):
    def __init__(self, root):
        self.root = root
        leaves_list = list()
        inode_list = list()
        root.fill_ml_node(leaves_list, inode_list)

        #叶节点
        leaf = list()
        #标签
        label = list()
        #非叶节点左右孩子
        treestr = list()
        #非叶节点
        treeopt = list()
        #非叶节点左右孩子门，如果只有一个孩子，第二个门等于0就行
        treestr_mask = list()

        for node in leaves_list:
            leaf.append(node.word)
            label.append(node.label)
        for node in inode_list:
            # leaf.append(node.idx)
            label.append(node.label)
            c = [node.children[0].idx, node.children[0].idx if len(node.children) < 2 else node.children[1].idx]
            treestr.append(c)
            treeopt.append(node.word)
            c = [1.0, 0.0 if len(node.children) < 2 else 1.0]
            treestr_mask.append(c)

        self.leaf = leaf
        self.label = label
        self.treestr = treestr
        self.treeopt = treeopt
        self.treestr_mask = treestr_mask
