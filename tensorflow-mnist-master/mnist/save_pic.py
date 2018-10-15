from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os
import numpy as np

#读取MNIST数据集，如果不存在会下载
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

#创建图片保存路径
save_dir = 'mnist_data/raw/'
#如果路径不存在，就创建此文件夹
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
#保存前20张图片
for i in range(20):
    # mnist.train.images[i, :] 表示第i张图片
    image_array = mnist.train.images[i, :]
    #将数据集中的784维的图片转化为28*28的图片
    image_array = image_array.reshape(28, 28)
    #将文件保存为jpg格式，命名为mnist_train_1.jpg,mnist_train_2.jpg等格式
    filename = save_dir + 'mnist_train_%d.jpg' % i
    #将image_array保存为图片，先用scipy.misc.toimage转换为图片再用save保存
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)
    print('图片label %s, label %s' % (mnist.train.labels[i, :], np.argmax(mnist.train.labels[i, :])))
