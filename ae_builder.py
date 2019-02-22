""" Autoencoder functions """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

#resolution = 32

class AE:
    def __init__(self, resolution, df,df_test, batch_start_train,batch_start_test,channels):

        self.resolution = resolution
        self.batch_start_train = batch_start_train
        self.batch_start_test = batch_start_test
        self.df = df
        self.df_test = df_test
        self.channels = channels
    def lrelu(self,x,alpha=0.1):
        return tf.maximum(alpha*x,x)

    def next_batch(self,df, size, t_set):

        if t_set == "train":
            batch = df[self.batch_start_train:self.batch_start_train+size]
            self.batch_start_train = self.batch_start_train + size - 1
        else:
            batch = df[0:10]
            self.batch_start_test = self.batch_start_test + size - 1
        return batch


    def model(self):
        inputs_ = tf.placeholder(tf.float32,[None,self.resolution,self.resolution,self.channels])
        targets_ = tf.placeholder(tf.float32,[None,self.resolution,self.resolution,self.channels])
            ### Encoder
        with tf.name_scope('en-convolutions'):
            conv1 = tf.layers.conv2d(inputs_,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=self.lrelu,name='conv1')
        # Now self.resolutionxself.resolutionx32
        with tf.name_scope('en-pooling'):
            maxpool1 = tf.layers.max_pooling2d(conv1,pool_size=(2,2),strides=(2,2),name='pool1')
        # Now 14x14x32
        with tf.name_scope('en-convolutions'):
            conv2 = tf.layers.conv2d(maxpool1,filters=32,kernel_size=(3,3),strides=(1,1),padding='SAME',use_bias=True,activation=self.lrelu,name='conv2')
        # Now 14x14x32
        with tf.name_scope('encoding'):
            encoded = tf.layers.max_pooling2d(conv2,pool_size=(2,2),strides=(2,2),name='encoding')
        # Now 7x7x32.
        #latent space
        ### Decoder
        with tf.name_scope('decoder'):
            conv3 = tf.layers.conv2d(encoded,filters=32,kernel_size=(3,3),strides=(1,1),name='conv3',padding='SAME',use_bias=True,activation=self.lrelu)
            #Now 7x7x32
            upsample1 = tf.layers.conv2d_transpose(conv3,filters=32,kernel_size=3,padding='same',strides=2,name='upsample1')
        # Now 14x14x32
            upsample2 = tf.layers.conv2d_transpose(upsample1,filters=32,kernel_size=3,padding='same',strides=2,name='upsample2')
        # Now self.resolutionxself.resolutionx32
            logits = tf.layers.conv2d(upsample2,filters=self.channels,kernel_size=(3,3),strides=(1,1),name='logits',padding='SAME',use_bias=True)
        #Now self.resolutionxself.resolutionx1
        # Pass logits through sigmoid to get reconstructed image
            decoded = tf.sigmoid(logits,name='recon')

        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=inputs_)
        #cost = tf.reduce_mean(tf.square(inputs_ - decoded))
        learning_rate=tf.placeholder(tf.float32)
        cost = tf.reduce_mean(loss)  #cost
        opt = tf.train.AdamOptimizer(learning_rate).minimize(cost) #optimizer
        # Training

        sess = tf.Session()
        #tf.reset_default_graph()

        saver = tf.train.Saver()
        loss = []
        valid_loss = []



        display_step = 1
        epochs = 30
        batch_size = 10
        #lr=[1e-3/(2**(i//5))for i in range(epochs)]
        lr=1e-5
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        for e in range(epochs):
            self.batch_start_train = 0
            self.batch_start_test = 0
            total_batch = int(len(self.df)/batch_size)
            for ibatch in range(total_batch):
                batch_x = self.next_batch(self.df,batch_size, "train")
                #batch_test_x= self.next_batch(self.df_test,10,"test")
                batch_x = np.asarray(batch_x)
                print(batch_x.shape)
                imgs_test = batch_x.reshape((-1, self.resolution, self.resolution, self.channels))
                noise_factor = 0.5
                #x_test_noisy = imgs_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs_test.shape)
                #x_test_noisy = np.clip(x_test_noisy, 0., 1.)
                imgs = batch_x.reshape((-1, self.resolution, self.resolution, self.channels))
                #x_train_noisy = imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)
                #x_train_noisy = np.clip(x_train_noisy, 0., 1.)
                batch_cost, _ = sess.run([cost, opt], feed_dict={inputs_: imgs,
                                                                 targets_: imgs,learning_rate:lr})

                batch_cost_test = sess.run(cost, feed_dict={inputs_: imgs,
                                                                 targets_: imgs_test})
            if (e+1) % display_step == 0:
                print("Epoch: {}/{}...".format(e+1, epochs),
                          "Training loss: {:.4f}".format(batch_cost),
                         "Validation loss: {:.4f}".format(batch_cost_test))

            loss.append(batch_cost)
            valid_loss.append(batch_cost_test)
            # plt.plot(range(e+1), loss, 'bo', label='Training loss')
            # plt.plot(range(e+1), valid_loss, 'r', label='Validation loss')
            # plt.title('Training and validation loss')
            # plt.xlabel('Epochs ',fontsize=16)
            # plt.ylabel('Loss',fontsize=16)
            # plt.legend()
            # plt.figure()
            # plt.show()
            saver.save(sess, './encode_model')


        batch_x = self.next_batch(self.df,10,"test")
        #imgs = batch_x[0].reshape((-1, self.resolution, self.resolution, 3))
        #for i in batch_x:
            #batch_x[i].reshape((-1, self.resolution, self.resolution, 3))
        imgs = np.asarray(batch_x)
        #noise_factor = 0.5
        #x_test_noisy = imgs + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=imgs.shape)
        #x_test_noisy = np.clip(x_test_noisy, 0., 1.)
        #x_test_noisy = x_test_noisy.reshape((-1, self.resolution, self.resolution, self.channels))
        imgs1 = imgs.reshape((-1, self.resolution, self.resolution,self.channels))
        recon_img = sess.run([decoded], feed_dict={inputs_: imgs1})[0]
        plt.figure(figsize=(20, 4))
        plt.title('Reconstructed Images')
        print("Original Images")
        for i in range(0,batch_size):
            plt.subplot(2, 10, i+1)
            plt.imshow(imgs[i], cmap = plt.cm.gray)
        plt.show()
        plt.figure(figsize=(20, 4))
        # print("Noisy Images")
        # for i in range(10):
        #     plt.subplot(2, 10, i+1)
        #     plt.imshow(x_test_noisy[i, ..., 0], cmap='gray')
        # plt.show()
        # plt.figure(figsize=(20, 4))

        print("Reconstruction of Noisy Images")
        print(recon_img.shape)
        for i in range(10):
            plt.subplot(2, 10, i+1)
            plt.imshow(recon_img[i], cmap='gray')
        plt.show()

        writer.close()

        sess.close()
