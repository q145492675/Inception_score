# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

# Call this function with list of images. Each of elements should be a 
# numpy array with values ranging from 0 to 255.
################################################################################
# 这其中images是CIFAR-10里面的真实数据（True，来源于Dataset），按照作者描述总共有三十张，
#保存在data这个文件下.
################################################################################
def get_inception_score(images, splits=10):
  assert(type(images) == list)
  assert(type(images[0]) == np.ndarray)
  assert(len(images[0].shape) == 3)
  assert(np.max(images[0]) > 10)
  assert(np.min(images[0]) >= 0.0)
  inps = []
  for img in images:
    img = img.astype(np.float32)
    inps.append(np.expand_dims(img, 0))
  bs = 100
################################################################################
#   这其中inp依然是真实数据， 然后sess我猜测是_init_inception()运行后的结果，那按照这个
#描述，pred是不是就是我们keras里面 model.predict()所产生的东西呢？ 我尝试print下来，发现
#pred是一个size为（30,1008）的数据，这个1008我查阅chainer(不是tensorflow)的人的代码，发现
#应该就是这个下面sess网络最后输出的维度了，那这个东西输出的目的是什么呢？
#  
#   Chainer 网络的URL(chainer这个程序的作者是基于这个tensorflow写的，他在里面定义了这个网络)
#所以通过这个网址我们能很好的明白这个sess究竟是长什么样子的. 
#  https://github.com/hvy/chainer-inception-score/blob/master/inception_score.py
################################################################################  
  with tf.Session() as sess:
    preds = []
    n_batches = int(math.ceil(float(len(inps)) / float(bs)))
    for i in range(n_batches):
        sys.stdout.write(".")
        sys.stdout.flush()
        inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
        inp = np.concatenate(inp, 0)
        pred = sess.run(softmax, {'ExpandDims:0': inp})
        print('the size: '+str(np.shape(pred)))
        preds.append(pred)       
    preds = np.concatenate(preds, 0)
    scores = []
################################################################################ 
# 这个片段就是具体来计算Inception_score了，但是问题就是它为什么要定义这个split，目的是
#为何。    
################################################################################    
    for i in range(splits):
      part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
      kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
      kl = np.mean(np.sum(kl, 1))
      scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


# This function is called automatically.
def _init_inception():
  global softmax
  if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
################################################################################
#   这一部分是从一个URL路径下载相应东西的程序， 这个程序我判断应该是一个神经网络训练完
#以后用tensorflow的形式来储存的结果。
################################################################################  
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
################################################################################  
#  这一模块，个人感觉是它在加载tensorflow里面一个叫做"graph"的属性，现在这个属性我还是
#不是很了解，我个人推测就和我们keras保存模型的h5ml文件，然后读取有点类似.
################################################################################  
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
  with tf.gfile.FastGFile(os.path.join(
      MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
  # Works with an arbitrary minibatch size. 
################################################################################
#  这下面的一个模块我个人认为会不会还是在读取网络的意思？ 它貌似创建了一个输出为nn.softmax
#的网络，这个网络的目的何在还不知道.
################################################################################  
  with tf.Session() as sess:
    pool3 = sess.graph.get_tensor_by_name('pool_3:0')
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            shape = [s.value for s in shape]
            new_shape = []
            for j, s in enumerate(shape):
                if s == 1 and j == 0:
                    new_shape.append(None)
                else:
                    new_shape.append(s)
            o._shape = tf.TensorShape(new_shape)
    w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
    print("\n The sieze of W :"+str(np.shape(w)))
    logits = tf.matmul(tf.squeeze(pool3), w)
    softmax = tf.nn.softmax(logits)
################################################################################
    
    
if __name__=='__main__':
    if softmax is None:
      _init_inception()
      
    def get_images(filename):
        return scipy.misc.imread(filename)
        
    filenames = glob.glob(os.path.join('./data', '*.*'))
    images = [get_images(filename) for filename in filenames]
#    print(len(images))
    print(get_inception_score(images))
