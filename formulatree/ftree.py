#coding=utf-8
#author=godpgf
from .ml import *

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


class FNode(object):

    def __init__(self, name, coff = None):
        self.name = name
        self.coff = coff
        self.children = []
        #父亲
        self.pre_id = -1
        #我是父亲的第几个孩子
        self.pre_child_index = -1

    def add_child(self, child_id):
        self.children.append(child_id)


class FTree(object):

    def __init__(self, node_list):
        self.node_list = node_list

    opt_map = {
        '+':"add",
        '-':"reduce",
        '*':"mul",
        '/':"div",
        '&':"and",
        '|':"or",
        '^':"signed_power",
        '<':"less",
        '>':"more",
        '?':"if",
        ':':"else(",
    }

    inv_opt_map = {
        "add":'+',
        "reduce":'-',
        "mul":'*',
        "div":'/',
        "and":'&',
        "or":'|',
        "signed_power":'^',
        "less":'<',
        "more":'>',
        "if":'?',
        "else":':',
    }

    cmp_set = set(["min", max])

    @classmethod
    def encode(cls, ftree):
        normal_line = []
        cls._encode(ftree.node_list, len(ftree.node_list)-1, normal_line)
        #print "".join(normal_line)
        return cls.fun2opt(normal_line)


    @classmethod
    def decode(cls, line):
        normal_line = cls.opt2fun(line)
        #print "".join(normal_line)
        node_list = []
        cls._decode(normal_line, 0, len(normal_line)-1, node_list)
        return FTree(node_list)

    #得到机器学习树
    def get_ml_tree_list(self, vocab):
        ml_tree_list = []

        #加入一个虚拟更节点，方便机器学习
        root = FNode('none')
        root.add_child(len(self.node_list) - 1)
        if len(self.node_list) > 0:
            self.node_list[-1].pre_id = len(self.node_list)
            self.node_list[-1].pre_child_index = 0
        self.node_list.append(root)

        for id, node in enumerate(self.node_list):
            if len(node.children) == 0:
                root = FTree.parse_ml_tree_from_leaf(self.node_list, id)
                MLNode.mend_ml_tree(root, vocab)
                ml_tree_list.append(MLTree(root))

        #删除之前的虚拟根节点
        self.node_list.pop()
        if len(self.node_list) > 0:
            self.node_list[-1].pre_id = -1
            self.node_list[-1].pre_child_index = -1

        return ml_tree_list


    @classmethod
    def get_vocab(cls, ftree_list):
        node_dict = {}
        coff_dict = {}
        for ftree in ftree_list:
            cls._get_all_node(len(ftree.node_list)-1, ftree.node_list, node_dict, coff_dict)
        return Vocab(node_dict, coff_dict)


    @classmethod
    def _get_all_node(cls, node_id, node_list, node_dict, coff_dict):
        node = node_list[node_id]
        if node.name in node_dict:
            assert len(node.children) == node_dict[node.name]
        else:
            node_dict[node.name] = len(node.children)
        if node.coff is not None:
            if node.name not in coff_dict:
                coff_dict[node.name] = [node.coff]
            else:
                coff_dict[node.name].append(node.coff)
        for child in node.children:
            cls._get_all_node(child, node_list, node_dict, coff_dict)

    @classmethod
    def _encode(cls, node_list, node_id, normal_line):
        node = node_list[node_id]
        #特殊处理系数在左边的情况
        if node.name.endswith("_from"):
            #先写名字
            normal_line.extend(list(node.name.replace("_from","")))
            normal_line.append('(')
            #写系数
            coff = str(node.coff)
            if coff.endswith(".0"):
                normal_line.extend(list(coff)[:-2])
            else:
                normal_line.extend(list(coff))
            #写左孩子
            normal_line.append(',')
            normal_line.append(' ')
            cls._encode(node_list, node.children[0], normal_line)
            normal_line.append(')')
        else:
            #先写名字
            normal_line.extend(list(node.name.replace("_to","")))
            if len(node.children) > 0:
                normal_line.append('(')
                cls._encode(node_list, node.children[0],normal_line)

                for i in range(1, len(node.children)):
                    normal_line.append(',')
                    normal_line.append(' ')
                    cls._encode(node_list, node.children[i], normal_line)

                if node.coff is not None:
                    normal_line.append(',')
                    normal_line.append(' ')
                    coff = str(node.coff)
                    if coff.endswith(".0"):
                        normal_line.extend(list(coff)[:-2])
                    else:
                        normal_line.extend(list(coff))
                normal_line.append(')')

    @classmethod
    def _decode(cls, normal_line, l, r, node_list):
        element = cls._decode_element(normal_line, l, r)
        opt = "".join(normal_line[element[0][0]:element[0][1]])

        #特殊处理一些操作符
        if opt in cls.inv_opt_map or opt in cls.cmp_set:
            #如果第一个孩子是数字, 变成opt_from
            if normal_line[element[1][0]] != '(' and is_number("".join(normal_line[element[1][0]:element[1][1]])):
                coff = float("".join(normal_line[element[1][0]:element[1][1]]))
                opt += "_from"
                node = FNode(opt, coff)
                node.add_child(cls._decode(normal_line, element[2][0], element[2][1]-1, node_list))
                node_list.append(node)
                for id, child in enumerate(node.children):
                    node_list[child].pre_id = len(node_list) - 1
                    node_list[child].pre_child_index = id
                return len(node_list) - 1
            #如果第二个数是数字，变成opt_to
            if normal_line[element[2][0]] != '(' and is_number("".join(normal_line[element[2][0]:element[2][1]])):
                opt += "_to"

        #特殊处理叶子节点
        if len(element) == 1:
            node = FNode(opt)
            node_list.append(node)
            return len(node_list) - 1

        coff = None
        if normal_line[element[-1][0]] != '(':
            data = "".join(normal_line[element[-1][0]:element[-1][1]])
            if is_number(data):
                coff = float(data)
                element.pop()
            elif "." in data and '(' not in data:
                coff = data
                element.pop()

        node = FNode(opt, coff)

        for i in range(1, len(element)):
            node.add_child(cls._decode(normal_line, element[i][0], element[i][1]-1, node_list))

        node_list.append(node)
        for id, child in enumerate(node.children):
            node_list[child].pre_id = len(node_list) - 1
            node_list[child].pre_child_index = id
        return len(node_list) - 1

    @classmethod
    def opt2fun(cls, line):
        line = list(line)
        normal_line = []
        i = 0
        while i < len(line):
            if line[i] in cls.opt_map and line[i+1] == ' ':
                depth = 1
                left_id = len(normal_line) - 1
                while depth != 0:
                    assert left_id >= 0
                    if normal_line[left_id] == '(':
                        depth -= 1
                    elif normal_line[left_id] == ')':
                        depth += 1
                    left_id -= 1

                #特殊处理":"，目的是将三个孩子变成两个
                if line[i] == ':':
                    left_id -= len(cls.opt_map['?'])

                tmp_l = normal_line[:left_id+1]
                tmp_r = normal_line[left_id+1:]
                normal_line = tmp_l
                #插入符号，特殊的如果是":"会多插入一个"("
                normal_line.extend(list(cls.opt_map[line[i]]))
                normal_line.extend(tmp_r)

                #删除多余空格
                while normal_line[-1] == ' ':
                    normal_line.pop()

                #特殊处理":",因为之前多插入一个"("，现在补回来
                if line[i] == ':':
                    normal_line.append(')')
                normal_line.append(',')
            else:
                normal_line.append(line[i])
            i += 1
        return "".join(normal_line)

    @classmethod
    def fun2opt(cls, normal_line):
        line = []
        cls._fun2opt(normal_line, line, 0, len(normal_line) - 1)
        return "".join(line)

    @classmethod
    def _fun2opt(cls, line, out_line, l, r):
        element = cls._decode_element(line, l, r)
        opt = "".join(line[element[0][0]:element[0][1]])
        if opt in cls.inv_opt_map:
            if cls.inv_opt_map[opt] != '?':
                out_line.append('(')
            cls._fun2opt(line, out_line, element[1][0], element[1][1] - 1)
            out_line.append(' ')
            out_line.append(cls.inv_opt_map[opt])
            out_line.append(' ')
            cls._fun2opt(line, out_line, element[2][0], element[2][1] - 1)
            if cls.inv_opt_map[opt] != '?':
                out_line.append(')')
        else:
            out_line.extend(list(opt))

            if len(element) > 1:
                out_line.append('(')
                cls._fun2opt(line, out_line, element[1][0], element[1][1] - 1)
                if len(element) > 2:
                    out_line.append(',')
                    out_line.append(' ')
                    cls._fun2opt(line, out_line, element[2][0], element[2][1] - 1)
                    if len(element) > 3:
                        out_line.append(',')
                        out_line.append(' ')
                        cls._fun2opt(line, out_line, element[3][0], element[3][1] - 1)
                out_line.append(')')

    #解码出函数名，左孩子，右孩子，系数
    @classmethod
    def _decode_element(cls, line, l, r):
        elememt = []
        #1、读出函数名
        depth = 0
        while line[l] == '(' or line[l] == ' ':
            if line[l] == '(':
                depth += 1
            l += 1
        opt_l = l
        while l <= r and line[l] != '(' and line[l] != ' ':
            l += 1
        #opt = line[opt_l:l]
        #print line[opt_l:l]
        #print line[opt_l:l]
        elememt.append((opt_l, l))
        #去掉左边括号
        if l > r:
            #全部读取完成
            assert depth == 0
            return elememt
        assert line[l] == '('
        l += 1
        depth += 1
        #去掉右边括号
        while l <= r and (line[r] == ')' or line[r] == ' '):
            if line[r] == ')':
                depth -= 1
            r -= 1
            #深度等于0时才可以结束
            assert r >= l
            if depth == 0:
                break

        #2、读出参数
        assert depth == 0
        id = l
        while id <= r:
            if line[id] == '(':
                depth += 1
            elif line[id] == ')':
                depth -= 1
            if depth == 0 and line[id] == ',':
                while line[l] == ' ':
                    l += 1
                elememt.append((l,id))
                #print line[l:id]
                l = id + 1

            #将最后一段做成孩子（不以','结尾)
            if id == r:
                assert depth == 0
                while line[l] == ' ':
                    l += 1
                elememt.append((l, r+1))
                #print line[l:r+1]

            id += 1
        return elememt

    # 每个alpha树的叶节点可以构成一个ml树根
    # last_node_index:-1表示没有上一个节点,0表示上个节点是父亲,1表示上个节点是左孩子,2表示上个节点是右孩子
    @classmethod
    def parse_ml_tree_from_leaf(cls, node_list, node_id, last_node_index=-1):
        # 最后一个叶节点不参与训练，仅作为标签
        root = node_list[node_id]
        if last_node_index == -1:
            if root.pre_id == -1:
                #只有一个节点
                assert root.name == 'none'
                return MLNode('none', 'none', [])
            return cls.parse_ml_tree_from_leaf(node_list, root.pre_id, 1 if root.pre_child_index == 0 else 2)

        child_list = list()
        if last_node_index == 0:
            for child in root.children:
                child_list.append(cls.parse_ml_tree_from_leaf(node_list, child, 0))
        elif last_node_index == 1 and root.pre_id != -1:
            child_list.append(cls.parse_ml_tree_from_leaf(node_list, root.pre_id, 1 if root.pre_child_index == 0 else 2))
        elif last_node_index == 2:
            if root.pre_id != -1:
                child_list.append(cls.parse_ml_tree_from_leaf(node_list, root.pre_id, 1 if root.pre_child_index == 0 else 2))
            child_list.append(cls.parse_ml_tree_from_leaf(node_list, root.children[0], 0))

        label = 'none'
        if last_node_index == 0:
            if root.pre_id != -1:
                label = node_list[root.pre_id].name
        elif last_node_index == 1:
            if len(root.children) > 0:
                label = node_list[root.children[0]].name
        else:
            if len(root.children) > 1:
                label = node_list[root.children[1]].name
        word = root.name
        if last_node_index == 0:
            if len(root.children) == 1:
                word = Vocab.get_down_2_up(word)
            elif len(root.children) == 2:
                word = Vocab.get_left_right_2_up(word)
        elif last_node_index == 2:
            word = Vocab.get_up_left_2_right(word)

        return MLNode(word, label, child_list)