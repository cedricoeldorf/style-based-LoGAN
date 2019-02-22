""" Autoencoder (Vanilla) """

import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data

# Get the MNIST data
#mnist = input_data.read_data_sets('./Data', one_hot=True)
from data_utilities import list_filenames, load_images
path = "./Data/LLD-logo_sample/"
filenames = list_filenames(path)

# All images within dataframe
resolution = 60
df = load_images(path, filenames, resolution,mnist = False)
del df[156]
df = np.array(df)
#df_test = df[-10:]
#df = df[0:-10]

# Parameters

input_dim = resolution*resolution
z_dim = 2
batch_size = 24
n_epochs = 20
learning_rate = 0.001
beta1 = 0.9
results_path = './experiment_2_results/Autoencoder'


# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[None,resolution,resolution,3], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[None,resolution,resolution,3], name='Target')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[None,15, 15,32], name='Decoder_input')

def lrelu(x,alpha=0.1):
    return tf.maximum(alpha*x,x)

def next_batch(df, batch_size,start):

    batch = df[start:start+batch_size]
    start = start + batch_size - 1
    return batch, start

def generate_image_grid(sess, df, filenames,resolution,batch_size,op, op2):
    """
    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
    :param sess: Tensorflow Session required to get the decoder output
    :param op: Operation that needs to be called inorder to get the decoder output
    :return: None, displays a matplotlib window with all the merged images.
    """

    nx, ny = 12, 1
    #plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=1, wspace=1)


    """ grid """
    input_x = sess.run(op2, feed_dict={x_input: df[0:batch_size]})
    for i, g in enumerate(gs):

        x = sess.run(op, feed_dict={decoder_input: input_x[i].reshape((-1,15,15,32))})
        ax = plt.subplot(g)
        print(x.shape)
        img = np.array(x).reshape(resolution,resolution,3)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        ax.set_title(filenames[i])
    plt.tight_layout()
    plt.show()

    for i, g in enumerate(gs):

        ax = plt.subplot(g)
        img = np.array(df[i])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
        ax.set_title(filenames[i])
    plt.tight_layout()
    plt.show()


def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_autoencoder". \
        format(datetime.datetime.now(), z_dim, learning_rate, batch_size, n_epochs, beta1)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The autoencoder network
def encoder(x, reuse=False):
    """
    Encode part of the autoencoder
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    def lrelu(x,alpha=0.1):
        return tf.maximum(alpha*x,x)
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        conv1 = tf.layers.conv2d(x,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv1')
        maxpool1 = tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),name='pool1')
        conv2 = tf.layers.conv2d(maxpool1,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=lrelu,name='conv2')
        encoded = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),name='encoding')
        #e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'e_dense_1'))
        #e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
        #latent_variable = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')
        return encoded


def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    def lrelu(x,alpha=0.1):
        return tf.maximum(alpha*x,x)
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        conv3 = tf.layers.conv2d(x,filters=32,kernel_size=(3,3),strides=(1,1),name='conv3',padding='SAME',use_bias=True,activation=lrelu)
        upsample1 = tf.layers.conv2d_transpose(conv3,filters=32,kernel_size=3,padding='same',strides=2,name='upsample1')
        upsample2 = tf.layers.conv2d_transpose(upsample1,filters=32,kernel_size=3,padding='same',strides=2,name='upsample2')
        logits = tf.layers.conv2d(upsample2,filters=3,kernel_size=(3,3),strides=(1,1),name='logits',padding='SAME',use_bias=True)
        decoded = tf.sigmoid(logits,name='recon')

        # d_dense_1 = tf.nn.relu(dense(x, z_dim, n_l2, 'd_dense_1'))
        # d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
        # output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        return decoded


def train(resolution,train_model):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
    :return: does not return anything
    """
    def lrelu(x,alpha=0.1):
        return tf.maximum(alpha*x,x)
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = encoder(x_input)
        decoder_output = decoder(encoder_output)

    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image = decoder(decoder_input, reuse=True)

    # Loss
    loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1).minimize(loss)
    init = tf.global_variables_initializer()

    # Visualization
    tf.summary.scalar(name='Loss', tensor=loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
    input_images = tf.reshape(x_input, [-1, resolution, resolution, 3])
    generated_images = tf.reshape(decoder_output, [-1, resolution, resolution, 3])
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
    summary_op = tf.summary.merge_all()

    # Saving the model
    saver = tf.train.Saver()
    step = 0
    start = 0
    with tf.Session() as sess:
        sess.run(init)
        if train_model:
            tensorboard_path, saved_model_path, log_path = form_results()
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            for i in range(n_epochs):
                n_batches = int(len(df) / batch_size)
                for b in range(n_batches):
                    #batch_x, _ = mnist.train.next_batch(batch_size)
                    """ Change 1: Data """
                    batch_x, start = next_batch(df, batch_size, start)

                    sess.run(optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    if b % 50 == 0:
                        batch_loss, summary = sess.run([loss, summary_op], feed_dict={x_input: batch_x, x_target: batch_x})
                        writer.add_summary(summary, global_step=step)
                        print("Loss: {}".format(batch_loss))
                        print("Epoch: {}, iteration: {}".format(i, b))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Loss: {}\n".format(batch_loss))
                    step += 1
                    """ Make start 0 again """
                    start = 0
                saver.save(sess, save_path=saved_model_path, global_step=step)
            print("Model Trained!")
            print("Tensorboard Path: {}".format(tensorboard_path))
            print("Log Path: {}".format(log_path + '/log.txt'))
            print("Saved Model Path: {}".format(saved_model_path))
        else:
            all_results = os.listdir(results_path)
            all_results.sort()
            saver.restore(sess,
                          save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))
            generate_image_grid(sess, df,filenames, resolution,batch_size,op=decoder_image, op2 = encoder_output)

if __name__ == '__main__':
    train(resolution,train_model=False)
