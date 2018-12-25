# coding=utf-8
# author=godpgf
import tensorflow as tf
from .tf_tree_lstm import TFTreeLSTM
from .data_utils import Vocab, MLNode, MLTree
import os

def create_model(vocab, batch_size=128, max_node_size=40):
    return TFTreeLSTM(vocab.one_child_index,
                       len(vocab.words) - vocab.one_child_index,
                       vocab.special_child_index,
                       batch_size=batch_size,
                       max_node_size=max_node_size
                       )

def train_model(sess, num_epochs, ml_tree_list, model):
    #tf.reset_default_graph()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(num_epochs):
        avg_loss = model.train(ml_tree_list, sess)
        if epoch % 10 == 0:
            print('epoch %d avg loss %.4f' % (epoch, avg_loss))

'''
def train(num_epochs, ml_tree_list, vocab, path="save/model.ckpt", batch_size=128, max_node_size=40):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        model = TFTreeLSTM(vocab.one_child_index,
                           len(vocab.words) - vocab.one_child_index,
                           vocab.special_child_index,
                           batch_size=batch_size,
                           max_node_size=max_node_size
                           )
        saver = tf.train.Saver()
        with tf.Session() as sess:
            fileName = path.split('/')[-1]
            if path and os.path.exists(path.replace(fileName, 'checkpoint')):
                saver.restore(sess, path)
            else:
                # init = tf.initialize_all_variables()
                init = tf.global_variables_initializer()
                sess.run(init)

            for epoch in range(num_epochs):
                avg_loss = model.train(ml_tree_list, sess)
                if epoch % 10 == 0:
                    print('epoch %d avg loss %.4f' % (epoch, avg_loss))
            saver.save(sess, path)
'''


def save_model(sess, path="save/model.ckpt"):
    saver = tf.train.Saver()
    saver.save(sess, path)


def load_model(sess, vocab, path="save/model.ckpt", batch_size=128, max_node_size=40):
    #tf.reset_default_graph()
    model = TFTreeLSTM(vocab.one_child_index,
                       len(vocab.words) - vocab.one_child_index,
                       vocab.special_child_index,
                       batch_size=batch_size,
                       max_node_size=max_node_size
                       )
    saver = tf.train.Saver()
    # sess = tf.Session()
    saver.restore(sess, path)
    return model
