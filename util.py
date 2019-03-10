import numpy as np
from matplotlib.pylab import scatter
import random
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from pylab import scatter
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import time
from pyntcloud import PyntCloud
import os
import sys
import itertools
import re

def stopWatch(value):
    valueD = (((value/365)/24)/60)
    Days = int (valueD)
    valueH = (valueD-Days)*365
    Hours = int(valueH)
    valueM = (valueH - Hours)*24
    Minutes = int(valueM)
    valueS = (valueM - Minutes)*60
    Seconds = int(valueS)
    print(Days,";",Hours,":",Minutes,";",Seconds)

def eukl_dis(a,b):
	a = np.array(a)
	b = np.array(b)
	dist = np.linalg.norm(a-b)
	return dist

def cal_inter(a,b):
	list_dic=[]
	index_list=[]
	for i in range(0,len(a)):
		list_shortes=[]
		for j in range(0,len(b)):
			dis = eukl_dis(a[i],b[j])
			list_shortes.append(dis)
		x = min(float(s) for s in list_shortes)
		index_min = min(range(len(list_shortes)), key=list_shortes.__getitem__)
		if x < 5:
			list_dic.append(x)
			index_list.append(index_min)

	return list_dic, index_list

def shuffle_data(training_data):
     np.random.shuffle(training_data)
     return training_data

def create_training_data():
	training_data = []
	print("Start loading training_data")
	#rotater = create_rotator()
	counter=0
	for k in range(0,439):
		cloud = PyntCloud.from_file("B%d.ply" % (k))
		cloud.add_scalar_field("hsv")
		voxelgrid_id = cloud.add_structure("voxelgrid", x_y_z=[430,430,430])
		points = cloud.get_sample("voxelgrid_centroids",voxelgrid_id=voxelgrid_id)
		new_cloud = PyntCloud(points)
		cloud1_test = new_cloud.get_sample(name="points_random",n = 2800)
		new_cloud = PyntCloud(cloud1_test)
		xyz_load = np.asarray(new_cloud.points)
		#print(xyz_load)
		training_data.append([xyz_load])
		new_cloud.to_file("out_file_%d.ply" % (k))
		print(k)
        #for r in range(0,len(rotater)):
            #training_data.append([])
            #for i in range(0,len(xyz_load)):
                #c = np.dot(xyz_load[i],rotater[r])
                #training_data[counter].append([c[0],c[1],c[2]])
            #counter = counter + 1
	print(len(training_data))
	print("data loaded")
	print("shuffle training data")
	training_data = np.asarray(training_data)
	print("getting Trainingdata into the right format")
	training_data = training_data.reshape(439,2800,3)
	print(" trainingdata formated")
	return training_data

def save_pointcloud(leaf,counter,leaf_name,number_points):
	leaf = np.asarray(leaf)
	leaf = np.reshape(leaf,(number_points,3))
	leaf_final = []
	x = 0
	for e in enumerate(leaf):
		leaf_final.append(tuple(leaf[x]))
		x = x +1
	vertex = np.array(leaf_final,dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
	el = PlyElement.describe(vertex, 'vertex')
	PlyData([el]).write('%s_%d.ply' % (leaf_name,counter))


def random_three_vector_cube():
	phi = np.random.uniform(0,np.pi*2)
	costheta = np.random.uniform(-2,2)
	theta = np.arccos( costheta )
	x = np.sin( theta) * np.cos( phi )
	y = np.sin( theta) * np.sin( phi )
	z = np.cos( theta )
	return (x,y,z)

def random_three_vector_sphere(radius):
	phi = np.random.uniform(0,np.pi*2)
	costheta = np.random.uniform(-1,1)
	u = np.random.uniform(0,1)
	theta =np.arccos( costheta )
	r = radius * np.cbrt( u )
	x = r * np.sin( theta) * np.cos( phi )
	y = r * np.sin( theta) * np.sin( phi )
	z = r * np.cos( theta )
	return (x,y,z)

def move_sphere(sphere,mover):
	for k in range(0,len(sphere)):
		sphere[k][0] = sphere[k][0] + mover[0]
		sphere[k][1] = sphere[k][1] + mover[1]
		sphere[k][2] = sphere[k][2] + mover[2]
	return sphere

def generate_sphere(number_points,radius):
	threetups = []
	for _ in range(number_points):
        	threetups.append(list(random_three_vector_sphere(radius)))
	fig = pyplot.figure()
	ax = Axes3D(fig)
	zipped = zip(*threetups)
	test = list(zipped)
	#ax.scatter(test[0],test[1],test[2])
	#pyplot.show()
	return threetups

def generate_cube(number_points,ratius):
	threetups = []
	for _ in range(5):
        	threetups.append(list(random_three_vector_cube()))
	#fig = pyplot.figure()
	#ax = Axes3D(fig)
	zipped = zip(*threetups)
	test = list(zipped)
	#ax.scatter(test[0],test[1],test[2])
	#pyplot.show()
	return test

def generate_all_sphere():
	mover = [[0,0,0],[0,0,5],[5,0,0],[0,5,0],[10,0,0],[0,10,0],[0,0,10],[20,0,0][0,20,0],[0,0,20],[30,0,0],[0,30,0],[0,0,30]]
	counter = 1
	sphere = generate_sphere(100000,5)
	save_pointcloud(sphere,counter,"sphere",len(sphere))
	counter +=1
	for i in range(0,len(mover)):
		sphere = move_sphere(sphere,mover[i])
		save_pointcloud(sphere,counter,"sphere",len(sphere))
		counter +=1
		#second_try = move_sphere(sphere,- 30.0)
		#save_pointcloud(second_try,3,"sphere",len(second_try))


def load_data(number_points,reduction_step):
	training_data = []
	#counter = 1
	for file in os.listdir("C:/Users/Andreas/Desktop/PG-PGGAN/table_new_%d" % (reduction_step)):
		if file.endswith(".ply"):
			cloud = PyntCloud.from_file("C:/Users/Andreas/Desktop/PG-PGGAN/table_new_%d/%s" % (reduction_step,file))
			cloud_array = np.asarray(cloud.points)
			training_data.append(cloud_array)
	return training_data

def load_data_table(number_points,reduction_step):
    training_data = []
    counter = 1
    if not os.path.exists("C:/Users/Andreas/Desktop/PG-PGGAN/table_new_%d" % reduction_step):
        os.mkdir("C:/Users/Andreas/Desktop/PG-PGGAN/table_new_%d" % reduction_step)
        table_uri = ("C:/Users/Andreas/Desktop/PG-PGGAN/table_new_%d" % reduction_step)
        print(table_uri)
        for file in os.listdir("C:/Users/Andreas/Desktop/PG-PGGAN/table"):
            if file.endswith(".ply"):
                cloud = PyntCloud.from_file("C:/Users/Andreas/Desktop/PG-PGGAN/table/%s" % file)
                cloud = cloud.get_sample(name="points_random",n = number_points)
                cloud = PyntCloud(cloud)
                cloud_array = np.asarray(cloud.points)
                cloud.to_file(table_uri + "/out_file_%d.ply" % (counter))
                counter = counter + 1
                training_data.append(cloud_array)

    else:
        training_data = load_data(number_points,reduction_step)
    print(len(training_data))
    print("data loaded")
    training_data = np.asarray(training_data)
    print(training_data.shape)
    print("getting Trainingdata into the right format")
    #training_data = training_data.reshape(8509,3072)
    print(training_data.shape)
    print(" trainingdata formated")
    return training_data


def generate_destroied_training_data():
    start = time.time()
    counter = 0
    print("reading training data")
    leafs = load_data()
    print("training data read")
    print("generate sphere")
    print("sphere generated")
    print(start)
    for i in range(349,350):
        string = ("destroyed_leaf%d" % (i))
        for j in range(0,45):
            mover = [[0,0,0],[0,0,8],[0,0,-8],[18,0,8],[-18,0,-8],[18,0,-8],[-18,0,8],[-18,0,0],[18,0,0],
            [0,15,0],[0,15,8],[0,30,-8],[18,15,8],[-18,15,-8],[18,15,-8],[-18,15,8],[-18,15,0],[18,15,0],
            [0,30,0],[0,30,8],[0,30,-8],[18,30,8],[-18,30,-8],[18,30,-8],[-18,30,8],[-18,30,0],[18,30,0],
            [0,-30,0],[0,-30,8],[0,-30,-8],[18,-30,8],[-18,-30,-8],[18,-30,-8],[-18,-30,8],[-18,-30,0],[18,-30,0],
            [0,-15,0],[0,-15,8],[0,-15,-8],[18,-15,8],[-18,-15,-8],[18,-15,-8],[-18,-15,8],[-18,-15,0],[18,-15,0]]
            sphere = generate_sphere(10000,10)
            sphere = move_sphere(sphere,mover[j])
            dif,index = cal_inter(sphere,leafs[i])
            index = set(index)
            index = list(index)
            le = np.array(leafs[i])
            for k in sorted(index,reverse=True):
                print(k)
                print(le.shape)
                le = np.delete(le,k,0)
                print(le.shape)
            finish = le
            save_pointcloud(finish,j,string,np.size(finish,0))
            print(counter)
            counter +=1
            end = time.time()
            print(end-start)

def reduce_dim_trainings_data_destroyed():
	training_data = []
	counter = 1
	for file in os.listdir("F:/punktwolkenplot/DC-PGAN/trainings_data_destroyed"):
		if file.endswith(".ply"):
			cloud = PyntCloud.from_file("F:/punktwolkenplot/DC-PGAN/trainings_data_destroyed/%s" % file)
			cloud = cloud.get_sample(name="points_random",n = 2048)
			cloud = PyntCloud(cloud)
			cloud_array = np.asarray(cloud.points)
			cloud.to_file("C:/Users/Andreas/Desktop/jupyter notebook/table/out_file_%d.ply" % (counter))
			counter = counter + 1
			training_data.append(cloud_array)
	print(len(training_data))
	print("data loaded")
	training_data = np.asarray(training_data)
	print(training_data.shape)
	print("getting Trainingdata into the right format")
	#training_data = training_data.reshape(8509,3072)
	print(training_data.shape)
	print(" trainingdata formated")
	return training_data

def reduce_dim_trainings_data():
	training_data = []
	counter = 1
	for file in os.listdir("F:/punktwolkenplot/DC-PGAN/trainings_data"):
		if file.endswith(".ply"):
			cloud = PyntCloud.from_file("F:/punktwolkenplot/DC-PGAN/trainings_data/%s" % file)
			cloud = cloud.get_sample(name="points_random",n = 2056)
			cloud = PyntCloud(cloud)
			cloud_array = np.asarray(cloud.points)
			#new_cloud.to_file("C:/Users/Andreas/Desktop/jupyter notebook/table/out_file_%d.ply" % (counter))
			counter = counter + 1
			training_data.append([cloud_array])
	print(len(training_data))
	print("data loaded")
	training_data = np.asarray(training_data)
	print(training_data.shape)
	print("getting Trainingdata into the right format")
	#training_data = training_data.reshape(8509,3072)
	print(training_data.shape)
	print(" trainingdata formated")
	return training_data



def load_data_destroyed():
    training_data = []
    for file in os.listdir("C:/Users/Andreas/Desktop/DC-PGAN/trainings_data_destroyed"):
        if file.endswith(".ply"):
            cloud = PyntCloud.from_file("C:/Users/Andreas/Desktop/DC-PGAN/trainings_data_destroyed/%s" % file)
            cloud_array = np.asarray(cloud.points)
            if cloud_array.shape[0] <= 2650 and cloud_array.shape[0]>=2048:
                cloud_new = cloud.get_sample(name="points_random",n = 2048)
                cloud_new = PyntCloud(cloud_new)
                cloud_array_new = np.asarray(cloud_new.points)
                if cloud_array_new.shape != (2048,6):
                    training_data.append(cloud_array_new)
                    cloud_new.to_file("C:/Users/Andreas/Desktop/DC-PGAN/trainings_data_destroyed_reduced/%s.ply" % (file))

    print(len(training_data))
    print("data loaded")
    training_data = np.asarray(training_data)
    print(training_data.shape)
    print("getting Trainingdata into the right format")
    #training_data = training_data.reshape(8509,3072)
    print(training_data.shape)
    print(" trainingdata formated")
    return training_data

def load_data_destroyed_reduced():
    training_data = []
    for file in os.listdir("C:/Users/Andreas/Desktop/DC-PGAN/trainings_data_destroyed_reduced"):
        if file.endswith(".ply"):
            cloud = PyntCloud.from_file("C:/Users/Andreas/Desktop/DC-PGAN/trainings_data_destroyed_reduced/%s" % file)
            cloud_array = np.asarray(cloud.points)
            if cloud_array.shape != (2056,6):
                training_data.append(cloud_array)


    print(len(training_data))
    print(training_data[0])
    print("data loaded")
    training_data = np.asarray(training_data)
    print(training_data.shape)
    print("getting Trainingdata into the right format")
    #training_data = training_data.reshape(8509,3072)
    print(training_data.shape)
    print(" trainingdata formated")
    return training_data

def load_data_all():
    # generate data for c gan for every destroyed_leaf get the right counterpart
    tester = load_data_destroyed_reduced()
    training_data = []
    all_data=[]
    for file in os.listdir("C:/Users/Andreas/Desktop/DC-PGAN/trainings_data_destroyed_reduced"):
        if file.endswith("ply"):
            k = int(re.search(r'\d+', file).group())
            print(k)
            training_data.append(k)
    training_data.sort()
    print(training_data)
    for i in range(0,len(training_data)):
        all_data.append(tester[training_data[i]])

    print(len(training_data))
    print(len(all_data))
    print(all_data[0][0][0])
    return all_data
''''
sum = 0
training_data = load_data(1028,4)
print(len(training_data))
for k in range(0,len(training_data)):
    counter = 0
    for i in range(0,len(training_data[k])):
        for j in range(0,len(training_data[k][i])):
            counter += training_data[k][i][j]
            sum += counter



print(sum/(len(training_data)*1028*3))
'''
