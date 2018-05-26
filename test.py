import sys
import numpy as np
import tensorflow as tf
import glob
from help_functions import *
from batch_generator import *
from CNN_building_blocks import *
from pc_processing import *
from read_files import *
from raw_to_grid import *


def test(pc,grid,grid_x,model,sess,resolution=0.2, is_velo_cam=False, \
			 scale=4, x=(0, 80), y=(-40, 40), z=(-2.5, 1.5)):

	objectness = model.objectness
	cordinate = model.cordinate
	y_pred = model.y
	
	objectness = sess.run(objectness, feed_dict={grid: grid_x})[0, :, :, :, 0]
	cordinate = sess.run(cordinate, feed_dict={grid: grid_x})[0]
	y_pred = sess.run(y_pred, feed_dict={grid: grid_x})[0, :, :, :, 0]
	print objectness.shape, objectness.max(), objectness.min()
	print y_pred.shape, y_pred.max(), y_pred.min()

	index = np.where(y_pred >= 0.995)
	print np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()
	print np.vstack((index[0], np.vstack((index[1], index[2])))).transpose().shape

	centers = np.vstack((index[0], np.vstack((index[1], index[2])))).transpose()
	centers = sphere_to_center(centers, resolution=resolution, \
		scale=scale, min_value=np.array([x[0], y[0], z[0]]))
	corners = cordinate[index].reshape(-1, 8, 3) + centers[:, np.newaxis]
	print corners.shape
	print grid.shape
	# publish_pc2(pc, corners.reshape(-1, 3))
	publish_pc2(pc, corners.reshape(-1, 3))
	# pred_corners = corners + pred_center
	# print pred_corners

def testing_batch(batch_size=5, resolution=0.1, dataformat="pcd", is_velo_cam=True, \
			scale=8, grid_shape=(800, 800, 40), x=(0, 80), y=(-40, 40), z=(-2.5, 1.5)):
	p = []
	#pc = None
	bounding_boxes = None
	places = None
	rotates = None
	size = None
	proj_velo = None

	for i in range(batch_size):
		tf.reset_default_graph()
		#k='%06d' %2397
		k='%010d' %i 
		pcd_path="/home/saichand/3D_CNN/2011_09_26/2011_09_26_drive_0013_sync/velodyne_points/data/"
		pcd_path=pcd_path+k+'.bin'

		if dataformat == "bin":
			pc = load_pc_from_bin(pcd_path)
		elif dataformat == "pcd":
			pc = load_pc_from_pcd(pcd_path)
		pc = filter_camera_angle(pc)

		grid =  raw_to_grid(pc, resolution=resolution, x=x, y=y, z=z)
		grid_x = grid.reshape(1, grid.shape[0], grid.shape[1], grid.shape[2], 1)
		print(grid_x.shape)

		with tf.Session() as sess:
			is_training=None
			model, grid, phase_train = grid_model(sess, grid_shape=grid_shape, activation=tf.nn.relu, is_training=is_training)
			saver = tf.train.Saver()
			new_saver = tf.train.import_meta_graph("velodyne_025_deconv_norm_valid40.ckpt.meta")
			last_model = "./velodyne_025_deconv_norm_valid40.ckpt"
			saver.restore(sess, last_model)
			test(pc=pc,grid=grid, grid_x=grid_x,model=model, sess=sess, resolution=0.1, is_velo_cam=True, scale=8, x=(0, 80),\
					y=(-40, 40), z=(-2.5, 1.5))
		sess.close()

if __name__ == '__main__':
	testing_batch(batch_size=15, resolution=0.1, dataformat="bin", is_velo_cam=True, \
			scale=8, grid_shape=(800, 800, 40), x=(0, 80), y=(-40, 40), z=(-2.5, 1.5))