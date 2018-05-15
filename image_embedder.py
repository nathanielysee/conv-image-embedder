import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datetime

filepath='/home/paperspace/Desktop/img_rec/rec_train_grouped_train/'
model_save_loc="/home/paperspace/Desktop/img_rec/model_saves/"
train_d=os.listdir('rec_train_grouped_train')
files = [filepath + i for i in train_d]

batch_size = 16
image_size = [256,256,3]

def parse_example(example, batch_size, image_size):
    '''convert tensor of strings from binaries into a tensor of images'''
    ex=tf.parse_example(example,
                        {'label':tf.FixedLenFeature([],tf.int64),
                         'image':tf.FixedLenFeature([],tf.string),
                         'img_id':tf.FixedLenFeature([],tf.string)
                        })
    return tf.reshape(tf.decode_raw(ex['image'],tf.uint8),[batch_size]+image_size)


def preprocess(inp,rot_range = 0.2,crop_margin = 0.2,sat_factor = 0.2,hue_factor = 0.2):
    '''apply random rotations, cropping and saturation and hue shifts to a batch of images'''
    images = tf.cast(inp, tf.float32)/255.
    shape = inp.shape.as_list()
    
    batch_size = shape[0]
    inds = tf.range(0, batch_size)
    img_shape = shape[1:3]

    x0y0 = tf.random_uniform([batch_size,2], 0, crop_margin)#upper left corners
    x1y1 = tf.random_uniform([batch_size,2], 1-crop_margin, 1)#lower right corners

    boxes = tf.concat([x0y0, x1y1], axis=1)#box for cropping image
    angles = tf.random_uniform([batch_size], -rot_range, rot_range)
    
    images = tf.contrib.image.rotate(images, angles)
    images = tf.image.crop_and_resize(images, boxes, inds,img_shape)
    
    images = tf.map_fn(
        lambda x: 
            tf.image.per_image_standardization(
            tf.image.random_saturation(
            tf.image.random_hue(x, hue_factor),1-sat_factor, 1+sat_factor)),
        images
    )
    
    return images


def pairwise_quadratic_loss(batch, diff):
    '''
    let r_ij be the distance between the i'th and j'th vector in a batch. 
    If batch corresponds to images of the same class, return sum of all r^2_ij.
    If batch corresponds to images of different classes, return sum of (r^2_ij - 4r_ij)
    '''
    batch_size = batch.shape.as_list()[0]
    
    reduce_sum_1 = tf.reduce_sum(tf.square(batch), 1) 
    
    term1 = tf.reduce_sum(reduce_sum_1, 0)*batch_size
    term2 = tf.reduce_sum(tf.square(tf.reduce_sum(batch,0)))
    squared_d = term1 - term2
    
    output=tf.cond(diff,
            lambda: squared_d - 4*tf.reduce_sum(tf.matrix_band_part(tf.sqrt(reduce_sum_1 + tf.transpose(reduce_sum_1) - 2*tf.matmul(batch,batch, transpose_b=True)+0.0001),0,-1)),
            lambda: squared_d
           )
    
    return output

def conv_embed(label, it_same, it_diff):

    next_same = it_same.get_next()
    next_diff = it_diff.get_next()

    input_layer = preprocess(tf.cond(label, lambda: next_diff, lambda: next_same))

    conv1 = tf.layers.conv2d(
        inputs = input_layer,
        filters = 64,
        kernel_size = [10,10],
        padding="valid",
        activation = tf.nn.relu
    )

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2,2], strides=2)
    
    conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 128,
        kernel_size = [7,7],
        padding="valid",
        activation = tf.nn.relu
    )
    
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size=[2,2], strides=2)
    
    conv3 = tf.layers.conv2d(
        inputs = pool2,
        filters = 128,
        kernel_size = [4,4],
        padding="valid",
        activation = tf.nn.relu
    )
    
    pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size=[2,2], strides=2)
    
    conv4 = tf.layers.conv2d(
        inputs = pool3,
        filters = 256,
        kernel_size = [4,4],
        padding="valid",
        activation = tf.nn.relu
    )
    
    pool4 = tf.layers.max_pooling2d(inputs = conv4, pool_size=[2,2], strides=2)
    
    conv5 = tf.layers.conv2d(
        inputs = pool4,
        filters = 256,
        kernel_size = [3,3],
        padding="valid",
        activation = tf.nn.relu
    )
    
    pool5 = tf.layers.max_pooling2d(inputs = conv5, pool_size=[2,2], strides=2)
    
    flat = tf.contrib.layers.flatten(inputs=pool5)
    
    dense = tf.layers.dense(inputs = flat, units=4096, activation=tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs=dense, rate=0.5)
    
    sphere = tf.nn.l2_normalize(dropout, axis=1)#project outputs of previous layer onto surface of hypersphere
    
    loss = pairwise_quadratic_loss(sphere, label)
    
    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    return {"embedding":sphere, "loss":loss, "train":train}

#create dataset where every image in a batch is of the same class
dataset_same = tf.data.Dataset.from_tensor_slices(files).interleave(lambda x: tf.data.TFRecordDataset(x).repeat().batch(batch_size).map(lambda x: parse_example(x, batch_size, image_size)),
                                                                    cycle_length = len(files),
                                                                    block_length = 1
                                                                   )
#create dataset where every image in a batch differs in class from every other image
dataset_diff = tf.data.Dataset.from_tensor_slices(files).interleave(lambda x: tf.data.TFRecordDataset(x).repeat(),
                                                                    cycle_length = len(files),
                                                                    block_length = 1
                                                                   ).batch(batch_size).map(lambda x:parse_example(x, batch_size, image_size))





it_same = dataset_same.make_one_shot_iterator()
it_diff = dataset_diff.make_one_shot_iterator()

saveable1 = tf.contrib.data.make_saveable_from_iterator(it_same)
saveable2 = tf.contrib.data.make_saveable_from_iterator(it_diff)

tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS,saveable1)
tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS,saveable2)

saver=tf.train.Saver()

label = tf.placeholder(tf.bool, shape=[])
model = conv_embed(label, it_same, it_diff)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


epoch_size = len(files)
epochs = 10

for i in range(epoch_size*epochs):#train model
    
    _, loss_value_same = sess.run((model["train"], model["loss"]), feed_dict={label:False})
    _, loss_value_diff = sess.run((model["train"], model["loss"]), feed_dict={label:True})
    print("{}".format(i))
    print("same: {}".format(loss_value_same))
    print("diff: {}".format(loss_value_diff))
    
    if not (i+1)%epoch_size:
        date_time = str(datetime.datetime.now()).split(".",1)[0].replace(" ", "_")
        savepath= model_save_loc + date_time + ".ckpt"
        saver.save(sess, savepath)
        print(date_time +": saved in " + savepath)
        

