import sys
import numpy as np
import tensorflow as tf
import glob
from help_functions import *
from CNN_building_blocks import *
from pc_processing import *
from read_files import *
from raw_to_grid import *

def batch_feed(batch_num, velodyne_path, label_path=None, calib_path=None, resolution=0.2, dataformat="pcd", label_type="txt", is_velo_cam=False, \
						scale=4, x=(0, 80), y=(-40, 40), z=(-2.5, 1.5)):
	velodynes_path = glob.glob(velodyne_path)
	labels_path = glob.glob(label_path) 
	calibs_path = glob.glob(calib_path)
	new_velodynes_path, new_labels_path, new_calibs_path = [], [], []
	for velodynes, labels, calibs in zip(velodynes_path, labels_path, calibs_path):
		proj_velo = None

		if calib_path:
			calib = read_calib_file(calibs)
			proj_velo = proj_to_velo(calib)[:, :3]

		if label_path:
			places, rotates, size = read_labels(labels, label_type, calib_path=calib_path, is_velo_cam=is_velo_cam, proj_velo=proj_velo)
			if places is not None:
				new_velodynes_path.append(velodynes)
				new_labels_path.append(labels)
				new_calibs_path.append(calibs)

	new_velodynes_path.sort()
	new_labels_path.sort()
	new_calibs_path.sort()


	iter_num = len(new_velodynes_path) // batch_num

	for itn in range(iter_num):
		batch_grid = []
		batch_g_map = []
		batch_g_cord = []

		for velodynes, labels, calibs in zip(new_velodynes_path[itn*batch_num:(itn+1)*batch_num], \
			new_labels_path[itn*batch_num:(itn+1)*batch_num], new_calibs_path[itn*batch_num:(itn+1)*batch_num]):
			p = []
			pc = None
			bounding_boxes = None
			places = None
			rotates = None
			size = None
			proj_velo = None

			if dataformat == "bin":
				pc = load_pc_from_bin(velodynes)
			elif dataformat == "pcd":
				pc = load_pc_from_pcd(velodynes)

			if calib_path:
				calib = read_calib_file(calibs)
				proj_velo = proj_to_velo(calib)[:, :3]

			if label_path:
				places, rotates, size = read_labels(labels, label_type, calib_path=calib_path, is_velo_cam=is_velo_cam, proj_velo=proj_velo)
				if places is None:
					continue

			corners = get_boxcorners(places, rotates, size)
			pc = filter_camera_angle(pc)

			grid =  raw_to_grid(pc, resolution=resolution, x=x, y=y, z=z)
			center_sphere, corner_label = create_label(places, size, corners, resolution=resolution, x=x, y=y, z=z, \
				scale=scale, min_value=np.array([x[0], y[0], z[0]]))

			if not center_sphere.shape[0]:
				print 1
				continue
			g_map = create_objectness_label(center_sphere, resolution=resolution, x=(x[1] - x[0]), y=(y[1] - y[0]), z=(z[1] - z[0]), scale=scale)
			g_cord = corner_label.reshape(corner_label.shape[0], -1)
			g_cord = corner_to_grid(grid.shape, g_cord, center_sphere, scale=scale)

			batch_grid.append(grid)
			batch_g_map.append(g_map)
			batch_g_cord.append(g_cord)
		yield np.array(batch_grid, dtype=np.float32)[:, :, :, :, np.newaxis], np.array(batch_g_map, dtype=np.float32), np.array(batch_g_cord, dtype=np.float32)

def train(batch_num, velodyne_path, label_path=None, calib_path=None, resolution=0.2, \
		dataformat="pcd", label_type="txt", is_velo_cam=False, scale=4, lr=0.01, \
		grid_shape=(800, 800, 40), x=(0, 80), y=(-40, 40), z=(-2.5, 1.5), epoch=101):
	# tf Graph input
	batch_size = batch_num
	training_epochs = epoch

	with tf.Session() as sess:
		model, grid, phase_train = grid_model(sess, grid_shape=grid_shape, activation=tf.nn.relu, is_training=True)
		saver = tf.train.Saver()
		total_loss, obj_loss, cord_loss, is_obj_loss, non_obj_loss, g_map, g_cord, y_pred = loss_function(model)
		optimizer = create_optimizer(total_loss, lr=lr)
		init = tf.global_variables_initializer()
		sess.run(init)

		for epoch in range(training_epochs):
			for (batch_x, batch_g_map, batch_g_cord) in batch_feed(batch_num, velodyne_path, label_path=label_path, \
			   calib_path=calib_path,resolution=resolution, dataformat=dataformat, label_type=label_type, is_velo_cam=is_velo_cam, \
			   scale=scale, x=x, y=y, z=z):
				sess.run(optimizer, feed_dict={grid: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train:True})
				cc = sess.run(cord_loss, feed_dict={grid: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train:True})
				iol = sess.run(is_obj_loss, feed_dict={grid: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train:True})
				nol = sess.run(non_obj_loss, feed_dict={grid: batch_x, g_map: batch_g_map, g_cord: batch_g_cord, phase_train:True})
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cc))
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(iol))
				print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(nol))
			if (epoch != 0) and (epoch % 1 == 0):
				print "Save epoch " + str(epoch)
				saver.save(sess, "./weights/velodyne" + str(epoch) + ".ckpt")
		print("Optimization Finished!")

if __name__ == '__main__':
	pcd_path = "/home/saichand/3D_CNN/data_object_velodyne/training/velodyne/*.bin"
	label_path = "/home/saichand/3D_CNN/label_2/*.txt"
	calib_path = "/home/saichand/3D_CNN/calib/*.txt"
	train(2, pcd_path, label_path=label_path, resolution=0.1, calib_path=calib_path, dataformat="bin", is_velo_cam=True, \
			 scale=8, grid_shape=(800, 800, 40), x=(0, 80), y=(-40, 40), z=(-2.5, 1.5))
		
	
	