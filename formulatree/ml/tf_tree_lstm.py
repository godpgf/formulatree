# coding=utf-8
# author=godpgf
# 在公式树中加入参数预测

import sys
import tensorflow as tf
import numpy as np


class TFTreeLSTM(object):
    def __init__(self,
                 num_emb,
                 num_opt_node,
                 output_dim,
                 emb_dim=8,
                 hidden_dim=6,
                 batch_size=128,
                 max_node_size=40,
                 lr=0.05,
                 emb_lr=0.1,
                 reg=0.0001
                 ):
        # 数据源数量，不包括操作符
        self.num_emb = num_emb
        # 动词数量，动词只能作为非叶节点
        self.num_opt_node = num_opt_node
        # 输出维度，即overmax行为数量
        self.output_dim = output_dim
        # 词向量维度
        self.emb_dim = emb_dim
        # 隐藏层数量
        self.hidden_dim = hidden_dim
        # 批处理数量
        self.batch_size = batch_size
        # 最大树的节点数量
        self.max_node_size = max_node_size

        self.reg = reg

        assert self.emb_dim > 1 and self.hidden_dim > 1

        self.add_placeholders()

        # 得到叶节点的词向量
        emb_leaves = self.add_embedding()

        self.add_model_variables()

        # 得到损失函数
        batch_loss = self.compute_loss(emb_leaves)

        self.loss, self.total_loss = self.calc_batch_loss(batch_loss)

        self.train_op_emb, self.train_op = self.add_training_op(lr, emb_lr)

    @classmethod
    def calc_wt_init(clf, fan_in=300):
        eps = 1.0 / np.sqrt(fan_in)
        return eps

    def add_placeholders(self):
        # 树的每个节点
        self.leaves = tf.placeholder(tf.int32, [self.batch_size, self.max_node_size], name='leaves')
        # 树的每个非叶节点对应的左右孩子（如果只有一个孩子，多出的就等于前面）
        self.treestr = tf.placeholder(tf.int32, [self.batch_size, self.max_node_size, 2], name='treestr')
        # 每个非叶节点的门，如只有左孩子，门就是[1.0,0.0]
        self.treestr_mask = tf.placeholder(tf.float32, [self.batch_size, self.max_node_size, 2], name='treestr_mask')
        # 树的每个非叶节点对应的内容
        self.treeopt = tf.placeholder(tf.int32, [self.batch_size, self.max_node_size], name='treeopt')
        # 树的每个节点对应的标签
        self.labels = tf.placeholder(tf.int32, [self.batch_size, self.max_node_size], name='labels')
        # 防止过拟合
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        # 当前批处理数量
        self.batch_len = tf.placeholder(tf.int32, name="batch_len")
        # 当前所有非叶节点数量
        self.n_inodes = tf.reduce_sum(tf.to_int32(tf.not_equal(self.treeopt, -1)), [1])
        # 当前叶节点数量，即数据源数量
        self.num_leaves = tf.reduce_sum(tf.to_int32(tf.not_equal(self.leaves, -1)), [1])

    def add_embedding(self):
        with tf.variable_scope("Embed", regularizer=None):
            # 将self.num_emb个整数映射到维度是self.emb_dim的矩阵
            embedding = tf.get_variable('embedding', [self.num_emb,
                                                      self.emb_dim]
                                        , initializer=tf.random_uniform_initializer(-0.05, 0.05), trainable=True,
                                        regularizer=None)
            # 将输入数据中多余的地方（等于-1的地方）填成0
            ix = tf.to_int32(tf.not_equal(self.leaves, -1)) * self.leaves
            # 所有输入数据全部映射（包括多余的地方，但他们的内容已经变成0）
            emb_tree = tf.nn.embedding_lookup(embedding, ix)
            # 将输入数据中多余的地方（等于-1的地方）和非叶节点都变成0，乘到刚才的映射（保证不需要的地方不会起作用）
            emb_tree = emb_tree * (tf.expand_dims(
                tf.to_float(tf.not_equal(self.leaves, -1)), 2))
            # 得到叶节点的词向量
            return emb_tree

    def add_model_variables(self):
        with tf.variable_scope("Composition", initializer=tf.contrib.layers.xavier_initializer(),
                               regularizer=tf.contrib.layers.l2_regularizer(self.reg)):
            # 用来将词向量映射到隐层的矩阵，包含short和long，所以输出维度是2*self.hidden_dim
            cW0 = tf.get_variable("cW0", [self.emb_dim, 2 * self.hidden_dim],
                                  initializer=tf.random_uniform_initializer(-self.calc_wt_init(), self.calc_wt_init()))
            cb0 = tf.get_variable("cb0", [2 * self.hidden_dim], initializer=tf.constant_initializer(0.0),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.0))

            # 动词的操作节点矩阵，degree表示有几个孩子
            degree = 2
            cW1 = tf.get_variable("cW1",
                                  [self.num_opt_node, degree * self.hidden_dim, (degree + 3) * self.hidden_dim],
                                  initializer=tf.random_uniform_initializer(-self.calc_wt_init(self.hidden_dim),
                                                                            self.calc_wt_init(self.hidden_dim)))
            cb1 = tf.get_variable("cb1", [self.num_opt_node, (degree + 2) * self.hidden_dim],
                                  initializer=tf.constant_initializer(0.0),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.0))


        with tf.variable_scope("Projection", regularizer=tf.contrib.layers.l2_regularizer(self.reg)):
            # 隐藏层到输出层
            U = tf.get_variable("U", [self.output_dim, self.hidden_dim],
                                initializer=tf.random_uniform_initializer(self.calc_wt_init(self.hidden_dim),
                                                                          self.calc_wt_init(self.hidden_dim)))
            bu = tf.get_variable("bu", [self.output_dim],
                                 initializer=tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))




    def compute_loss(self, emb_batch):
        outloss = []
        prediction = []
        for idx_batch in range(self.batch_size):
            # 读出当前批次的所有序列隐藏层的输出
            tree_states = self.compute_states(emb_batch, idx_batch)
            # 得到当前批次的所有输出节点的输出
            logits = self.create_output(tree_states)
            # 得到当前批次的标签
            labels1 = tf.gather(self.labels, idx_batch)
            labels_num = tf.reduce_sum(tf.to_int32(tf.not_equal(labels1, -1)))
            labels = tf.gather(labels1, tf.range(labels_num))
            # 得到这个批次的损失函数
            label_loss = self.calc_label_loss(logits, labels)
            # 得到这个批次的最后一个预测
            pred = tf.nn.softmax(logits)
            pred_root = tf.gather(pred, labels_num - 1)
            prediction.append(pred_root)
            outloss.append(label_loss)


        # 得到所有批次的损失和最后一个预测
        batch_loss = tf.stack(outloss)
        self.pred = tf.stack(prediction)

        return batch_loss

    # 读出当前批次下通过lstm得到的每个隐藏层节点
    def compute_states(self, emb, idx_batch):
        # 读出当前批次下叶节点
        num_leaves = tf.squeeze(tf.gather(self.num_leaves, idx_batch))
        # num_leaves=tf.Print(num_leaves,[num_leaves])
        # 读出当前批次非叶节点
        n_inodes = tf.gather(self.n_inodes, idx_batch)
        # 取出当前批次叶节点的词向量
        embx = tf.gather(tf.gather(emb, idx_batch), tf.range(num_leaves))
        # 取出当前批次非叶节点的左右孩子id
        treestr = tf.gather(tf.gather(self.treestr, idx_batch), tf.range(n_inodes))
        # 取出当前批次非叶节点的左右孩子门
        treestr_mask = tf.gather(tf.gather(self.treestr_mask, idx_batch), tf.range(n_inodes))
        # 取出当前批次非叶节点的操作符
        treeopt = tf.gather(tf.gather(self.treeopt, idx_batch), tf.range(n_inodes))

        # 取出叶节点的长短序列
        leaf_hc = self.process_leafs(embx)
        leaf_h, leaf_c = tf.split(leaf_hc, 2, 1)
        # 创建一个新的图，相当于node_h=leaf_h*1,node_c=leaf_c*1
        node_h = tf.identity(leaf_h)
        node_c = tf.identity(leaf_c)

        # 创建下标，用来循环遍历所有的非叶节点
        idx_var = tf.constant(0)  # tf.Variable(0,trainable=False)

        with tf.variable_scope("Composition", reuse=True):
            # 动词的矩阵
            degree = 2
            cW1 = tf.get_variable("cW1", [self.num_opt_node, degree * self.hidden_dim, (degree + 3) * self.hidden_dim])
            cb1 = tf.get_variable("cb1", [self.num_opt_node, (degree + 2) * self.hidden_dim])

            # 遍历每个非叶节点
            def _recurrence(node_h, node_c, idx_var):
                # 得到当前节点的孩子节点对应的下标，如[3,5]
                node_info = tf.gather(treestr, idx_var)
                # 得到当前节点的孩子节点对应的下标门
                node_info_mask = tf.gather(treestr_mask, idx_var)

                # 取出孩子节点的长短序列
                child_h = tf.gather(node_h, node_info)
                child_c = tf.gather(node_c, node_info)

                #使用门
                child_h_l, child_h_r = tf.split(child_h,2)
                child_c_l, child_c_r = tf.split(child_c,2)
                child_h = tf.concat([tf.gather(node_info_mask,0) * child_h_l, tf.gather(node_info_mask,1) * child_h_r], 0)
                child_c = tf.concat([tf.gather(node_info_mask,0) * child_c_l, tf.gather(node_info_mask,1) * child_c_r], 0)

                flat_ = tf.reshape(child_h, [-1])

                # 取出当前节点的矩阵下标
                node_opt = tf.gather(treeopt, idx_var) - self.num_emb
                cW = tf.gather(cW1, node_opt)
                cb = tf.gather(cb1, node_opt)

                # 根据lstm公式，输入短序列，得到一些中间cell
                # f是长序列的门，用来决定保留哪些长序列
                # i是短序列的精英，用来添加到现有的长序列上
                # u是短序列的门，用来决定保留哪些短序列的精英到长序列上
                # o是下一个短序列
                tmp = tf.matmul(tf.expand_dims(flat_, 0), cW)
                u, o, i, fl, fr = tf.split(tmp, 5, 1)

                # 因为fr和fl会加到一起作为长序列的门，所以bf可以被他们两共用
                bu, bo, bi, bf = tf.split(cb, 4, 0)

                i = tf.nn.sigmoid(i + bi)
                o = tf.nn.sigmoid(o + bo)
                u = tf.nn.tanh(u + bu)
                fl = tf.nn.sigmoid(fl + bf)
                fr = tf.nn.sigmoid(fr + bf)

                f = tf.concat([fl, fr], 0)

                # 计算出下一个长短序列
                c = i * u + tf.reduce_sum(f * child_c, [0])
                h = o * tf.nn.tanh(c)

                # 将得到的新长短序列放到末尾
                node_h = tf.concat([node_h, h], 0)
                node_c = tf.concat([node_c, c], 0)

                idx_var = tf.add(idx_var, 1)

                return node_h, node_c, idx_var

            loop_cond = lambda a1, b1, idx_var: tf.less(idx_var, n_inodes)

            loop_vars = [node_h, node_c, idx_var]
            node_h, node_c, idx_var = tf.while_loop(loop_cond, _recurrence,
                                                    loop_vars, parallel_iterations=10)

            return node_h

    # 返回叶节点的短长序列hc
    def process_leafs(self, emb):

        with tf.variable_scope("Composition", reuse=True):
            cW0 = tf.get_variable("cW0", [self.emb_dim, 2 * self.hidden_dim])
            cb0 = tf.get_variable("cb0", [2 * self.hidden_dim])

            def _recurseleaf(x):
                # 先乘上一个矩阵，得到u=c[t-1],o=h[t-1]
                concat_uo = tf.matmul(tf.expand_dims(x, 0), cW0) + cb0
                u, o = tf.split(concat_uo, 2, 1)

                # 得到短序列的输出，输出的内容由长序列的门来决定
                o = tf.nn.sigmoid(o)
                c = tf.nn.tanh(u)
                # 用门来控制短序列的输出
                h = o * tf.nn.tanh(c)

                hc = tf.concat([h, c], 1)
                hc = tf.squeeze(hc)
                return hc

        hc = tf.map_fn(_recurseleaf, emb)
        return hc

    # 得到隐藏层每个节点的输出
    def create_output(self, tree_states):

        with tf.variable_scope("Projection", reuse=True):
            U = tf.get_variable("U", [self.output_dim, self.hidden_dim],
                                )
            bu = tf.get_variable("bu", [self.output_dim])

            h = tf.matmul(tree_states, U, transpose_b=True) + bu
            return h

    def calc_label_loss(self, logits, labels):
        # 交叉熵损失函数
        l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(None,
                                                            labels, logits)
        loss = tf.reduce_sum(l1, [0])
        return loss

    #def calc_coff_label_loss(self, coff_values, coff_labels):
    #    #用最最简单的线性回归
    #    l1 = tf.square(coff_values - coff_labels)
    #    loss = tf.reduce_mean(l1, [0])
    #    return loss

    # 收集回归的损失
    def calc_batch_loss(self, batch_loss):
        # 收集所有层输出的损失（而不仅仅是最后一层）
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # 把所有输入的tensor按照元素相加
        regpart = tf.add_n(reg_losses)
        loss = tf.reduce_mean(batch_loss)
        total_loss = loss + 0.5 * regpart
        return loss, total_loss

    def add_training_op(self, lr, emb_lr):
        loss = self.total_loss
        # 创建优化器，传入学习率，他们是用来最小化tensor的
        opt = tf.train.AdagradOptimizer(lr)
        optemb = tf.train.AdagradOptimizer(emb_lr)

        # 返回图中所有训练参数的名字
        ts = tf.trainable_variables()
        for t in ts:
            print(t)

        gs = tf.gradients(loss, ts)
        gs_ts = zip(gs, ts)

        gt_emb, gt_nn = [], []
        for g, t in gs_ts:
            # print t.name,g.name
            if "Embed/embedding:0" in t.name:
                gt_emb.append((g, t))
            else:
                gt_nn.append((g, t))

        train_op = opt.apply_gradients(gt_nn)
        train_op_emb = optemb.apply_gradients(gt_emb)
        train_op = [train_op_emb, train_op]

        return train_op

    def train(self, ml_tree_list, sess):
        from random import shuffle
        data_idxs = [i for i in range(len(ml_tree_list))]
        shuffle(data_idxs)
        losses = []
        for i in range(0, len(ml_tree_list), self.batch_size):
            batch_size = min(i + self.batch_size, len(ml_tree_list)) - i
            # 不到batch_size的不参与训练，下一个epoch又会随机到的，所以没关系
            if batch_size < self.batch_size: break

            batch_idxs = data_idxs[i:i + batch_size]
            batch_data = [ml_tree_list[ix] for ix in batch_idxs]  # [i:i+batch_size]

            leaf_b, treestr_b, treestr_mask_b, treeopt_b, labels_b = self.extract_batch_tree_data(
                batch_data, self.max_node_size)
            feed = {
                self.leaves: leaf_b,
                self.treestr: treestr_b,
                self.treestr_mask: treestr_mask_b,
                self.treeopt: treeopt_b,
                self.labels: labels_b
            }

            loss, _, _= sess.run(
                    [self.loss, self.train_op_emb, self.train_op], feed_dict=feed)
            losses.append(loss)
            sstr = 'avg loss %.2f at example %d of %d\r' % (loss, i, len(ml_tree_list))
            sys.stdout.write(sstr)
            sys.stdout.flush()
        return np.mean(losses)

    def evaluate(self, ml_tree_list, sess, is_test=False):
        #num_correct = 0
        #total_data = 0
        data_idxs = range(len(ml_tree_list))
        test_batch_size = self.batch_size
        predicts = list()
        for i in range(0, len(ml_tree_list), test_batch_size):
            batch_size = min(i + test_batch_size, len(ml_tree_list)) - i
            if batch_size < test_batch_size: break
            batch_idxs = data_idxs[i:i + batch_size]
            batch_data = [ml_tree_list[ix] for ix in batch_idxs]  # [i:i+batch_size]

            leaf_b, treestr_b, treestr_mask_b, treeopt_b, labels_b = self.extract_batch_tree_data(
                batch_data, self.max_node_size)
            feed = {
                self.leaves: leaf_b,
                self.treestr: treestr_b,
                self.treestr_mask: treestr_mask_b,
                self.treeopt: treeopt_b,
                self.labels: labels_b,
                self.dropout: 1.0,
                self.batch_len: len(leaf_b)
            }
            #labels_root = [l[-1] for l in labels_b]

            pred_y = sess.run(self.pred, feed_dict=feed)
            # break
            predicts.append(pred_y)
        return predicts

    @classmethod
    def extract_batch_tree_data(cls, batchdata, fillnum=120):

        dim1, dim2 = len(batchdata), fillnum
        # 叶节点
        leaf_emb_arr = np.empty([dim1, dim2], dtype='int32')
        leaf_emb_arr.fill(-1)
        # 非叶节点的孩子
        treestr_arr = np.empty([dim1, dim2, 2], dtype='int32')
        treestr_arr.fill(-1)
        treestr_mast_arr = np.empty([dim1, dim2, 2], dtype='float')
        treestr_mast_arr.fill(0.0)
        # 非叶节点的操作
        treeopt_arr = np.empty([dim1, dim2], dtype='int32')
        treeopt_arr.fill(-1)
        # 标签
        labels_arr = np.empty([dim1, dim2], dtype=float)
        labels_arr.fill(-1)
        for i, tree in enumerate(batchdata):
            leaf, label, treestr, treeopt, treestr_mask = tree.leaf, tree.label, tree.treestr, tree.treeopt, tree.treestr_mask
            leaf_emb_arr[i, 0:len(leaf)] = leaf
            if len(treestr) > 0:
                treestr_arr[i, 0:len(treestr), 0:2] = treestr
                treestr_mast_arr[i, 0:len(treestr_mask), 0:2] = treestr_mask
            treeopt_arr[i, 0:len(treeopt)] = treeopt
            labels_arr[i, 0:len(label)] = label

        return leaf_emb_arr, treestr_arr, treestr_mast_arr, treeopt_arr, labels_arr
