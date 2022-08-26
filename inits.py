import tensorflow as tf
import numpy as np
# DISCLAIMER:
# This file is derived from 
# https://github.com/tkipf/gcn
# which is also under the MIT license

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    # 在该[minval, maxval)范围内生成均匀分布的数值
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    # shape[0]+shape[1]分别为权重张量的输入单元数和输出单元数
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
