import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from pcpggan import PCPGGAN
#from helper import create_training_data
import sys

flags = tf.app.flags


flags.DEFINE_integer("epochs",100,"epochs per trainingstep")
flags.DEFINE_float("learning_rate",0.0001,"learning rate for the model")
flags.DEFINE_integer("z_dim",126,"dimension of the noise Input")
flags.DEFINE_bool("training",True,"running training of the poincloud gan")
flags.DEFINE_string("checkpoint_dir","C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/","where to save the model")
flags.DEFINE_string("sample_dir","samples","where the samples are stored")
flags.DEFINE_integer("number_point",2048,"number of points in each pointcloud")
flags.DEFINE_integer("pointcloud_dim",2048,"number of input")
flags.DEFINE_integer("iterations",100000,"number of patches")
flags.DEFINE_integer("batch_size",32,"size of the batch")
flags.DEFINE_float("beta1",0.5,"adam beta1")
flags.DEFINE_float("beta2",0.5,"adam beta2")
flags.DEFINE_integer("growth",10,"number of grow steps of the progressive growing gan")
FLAGS = flags.FLAGS



growth = [128,256,512,1028,2032]
def _main(argv):
	print("initializing Params")
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)
	if FLAGS.training == True:
		for i in range(1,(len(growth)+1)):
			print("building the Model Stage_%d" % (i))
			print(growth[0:i])
			print(i)
			pcpggan = PCPGGAN(FLAGS.training,FLAGS.epochs,growth[i-1],FLAGS.checkpoint_dir,FLAGS.learning_rate,FLAGS.z_dim,FLAGS.batch_size,FLAGS.beta1,FLAGS.beta2,growth[0:i],i)
			pcpggan.train()
	else:
		if not pgpggan.load(FLAGS.checkpoint_dir):
			print("first train your model")

if __name__ == '__main__':
	print('Starting the Programm....')
	app.run(_main)
