import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from util import load_data_table, save_pointcloud, shuffle_data
import time
import math
from random import shuffle
from pyntcloud import PyntCloud
import trimesh
import pandas as pd
from plyfile import PlyData, PlyElement
import random

class PCPGGAN(object):

	def __init__(self,is_training,epoch,pointcloud_dim,checkpoint_dir, learning_rate,z_dim,batch_size,beta1,beta2,growth,number_prog):
		self.beta1 = beta1
		self.beta2 = beta2
		self.learning_rate = learning_rate
		self.z_dim = z_dim
		self.pointcloud_dim = pointcloud_dim
		self.training = is_training
		self.batch_size = batch_size
		self.epoch = epoch
		self.checkpoint_dir = checkpoint_dir
		self.table_fake = "table_fake"
		self.growth = growth
		self.number_prog = number_prog
		self.lam_gp = 10
		self.build_network()
		print("start training")
		print("d_optim")
		with tf.variable_scope("adam",reuse=tf.AUTO_REUSE) as scope:
			self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.g_loss,var_list = self.vars_G)
			self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1).minimize(self.d_loss,var_list = self.vars_D)
		print("g_optim")
		self.init  = tf.global_variables_initializer()
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True


	def generator(self,x,growth):
		with tf.variable_scope("generator",reuse=tf.AUTO_REUSE) as scope:
			for i in range(1):
				x = tf.layers.dense(x, growth[-1]*3, activation=tf.nn.leaky_relu, name = "gen_layer_%d" % self.number_prog,reuse=tf.AUTO_REUSE)
				x = tf.layers.dense(x, growth[-1]*3, activation=None,kernel_initializer = tf.constant_initializer(0.2), name="gen_layer_last%d" % self.number_prog,reuse=tf.AUTO_REUSE)
			x = tf.reshape(x,[-1,growth[-1],3])

		return x

	def discriminator(self,y,growth,reuse=False):
		with tf.variable_scope("discriminator") as scope:
			if reuse:
				scope.reuse_variables()
			for i in range(1):
				y = tf.layers.conv1d(y,filters = 64,kernel_size=3,strides=1,padding="same",activation=tf.nn.leaky_relu,name = "conv_%d" % (self.number_prog),reuse=tf.AUTO_REUSE)
			y = tf.layers.dense(y, 128,activation=tf.nn.leaky_relu, name="conv_1last",reuse=tf.AUTO_REUSE)
			y = tf.layers.dense(y, 1, activation=None,name="conv_last",reuse=tf.AUTO_REUSE)

		return y

	def build_network(self):
		eps = 1e-12
		self.input = tf.placeholder(tf.float32, [None,self.growth[-1],3], name="real_pointcloud_data")
		self.z = tf.placeholder(tf.float32,[None,self.z_dim], name ="noice")
		self.Gen = self.generator(self.z,self.growth)
		self.Dis_real = self.discriminator(self.input,self.growth,reuse = False)
		self.Dis_fake = self.discriminator(self.Gen,self.growth,reuse = True)

		#Tensorboard variables
		self.d_sum_real = tf.summary.histogram("d_real", self.Dis_real)
		self.d_sum_fake = tf.summary.histogram("d_fake", self.Dis_fake)
		self.G_sum = tf.summary.histogram("G",self.Gen)
		self.z_sum = tf.summary.histogram("z_input",self.z)

		#Wassersteinmetrik
		self.d_loss = tf.reduce_mean(self.Dis_fake - self.Dis_real)
		self.g_loss = -tf.reduce_mean(self.Dis_fake)

		# gradient penalty
		self.dif = self.Gen - self.input
		self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
		self.interpoltate = self.input + (self.alpha * self.dif)
		discre_logits = self.discriminator(self.interpolate,self.growth,reuse = True)
		gradients = tf.gradients(discri_logits, [interpolates])[0]

		slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

		self.d_loss += self.lam_gp * self.gradient_penalty

		#Vanilla GAN
		#self.d_loss = tf.reduce_mean(-tf.log(self.Dis_real) - tf.log(1. - self.Dis_fake))
		#self.g_loss = tf.reduce_mean(-tf.log(self.Dis_fake))
		tf.summary.scalar('self.g_loss', self.g_loss )
		tf.summary.scalar('self.d_loss', self.d_loss )

		t_vars = tf.trainable_variables()
		self.vars_D = [var for var in t_vars if 'conv' in var.name]
		self.vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
		self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
		self.saver = tf.train.Saver()

		#Tensorboard variables
		self.summary_g_loss = tf.summary.scalar("g_loss",self.g_loss)
		self.summary_d_loss = tf.summary.scalar("d_loss",self.d_loss)

	def train(self):

		with tf.Session(config = self.config) as sess:
			#imported_meta = tf.train.import_meta_graph("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4.meta")
			#imported_meta.restore(sess, "C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4")

			train_writer = tf.summary.FileWriter("./logs",sess.graph)
			merged = tf.summary.merge_all()
			#test_writer = tf.summary.FileWriter("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/")
			self.counter = 1
			sess.run(self.init)
			self.training_data = load_data_table(self.growth[-1],len(self.growth))
			k = (len(self.training_data) // self.batch_size)
			self.start_time = time.time()
			loss_g_val,loss_d_val = 0, 0
			self.training_data = self.training_data[0:(self.batch_size*k)]

			print("Lengh of the training_data:")
			print(len(self.training_data))
			for e in range(0,self.epoch):
				epoch_loss_d = 0.
				epoch_loss_g = 0.
				print("data shuffeld")
				self.training_data = shuffle_data(self.training_data)

				for i in range(0,k):
					self.batch_z = np.random.uniform(0, 0.2, [self.batch_size, self.z_dim])
					self.batch = self.training_data[i*self.batch_size:(i+1)*self.batch_size]
					_, loss_d_val,loss_d = sess.run([self.d_optim,self.d_loss,self.summary_d_loss],feed_dict={self.input: self.batch,self.z: self.batch_z})
					train_writer.add_summary(loss_d,self.counter)
					_, loss_g_val,loss_g = sess.run([self.g_optim,self.g_loss,self.summary_g_loss],feed_dict={self.z: self.batch_z})
					train_writer.add_summary(loss_g,self.counter)
					self.counter=self.counter + 1
					epoch_loss_d += loss_d_val
					epoch_loss_g += loss_g_val
				epoch_loss_d /= k
				epoch_loss_g /= k
				print("Loss of D: %f" % epoch_loss_d)
				print("Loss of G: %f" % epoch_loss_g)
				print("Epoch%d" %(e))

				if e % 100 == 0:
					#save_path = self.saver.save(sess,"C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt",global_step=e)
					#print("model saved: %s" %save_path)
					self.gen_noise = np.random.uniform(0, 0.2, [1, self.z_dim])
					test_table = sess.run([self.Gen], feed_dict={self.z: self.gen_noise})
					save_pointcloud(test_table,self.number_prog,self.table_fake, self.growth[-1])
					print("created fake_test_leaf")

			print("training finished")
