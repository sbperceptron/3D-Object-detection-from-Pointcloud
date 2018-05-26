import tensorflow as tf


def batch_norm(inputs, phase_train, decay=0.9, eps=1e-5):
	
	gamma = tf.get_variable("gamma", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
	beta = tf.get_variable("beta", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
	pop_mean = tf.get_variable("pop_mean", trainable=False, shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
	pop_var = tf.get_variable("pop_var", trainable=False, shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
	axes = range(len(inputs.get_shape()) - 1)

	if phase_train != None:
		batch_mean, batch_var = tf.nn.moments(inputs, axes)
		train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean*(1 - decay))
		train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, eps)
	else:
		return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, gamma, eps)


def conv3DLayer(input_layer, input_dim, output_dim, height, width, length, stride, activation=tf.nn.relu, padding="SAME", name="", is_training=True):
	with tf.variable_scope("conv3D" + name):
		kernel = tf.get_variable("weights", shape=[length, height, width, input_dim, output_dim], \
			dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
		b = tf.get_variable("bias", shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		conv = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)
		bias = tf.nn.bias_add(conv, b)
		if activation:
			bias = activation(bias, name="activation")
		bias = batch_norm(bias, is_training)
	return bias

def conv3D_to_output(input_layer, input_dim, output_dim, height, width, length, stride, activation=tf.nn.relu, padding="SAME", name=""):
	with tf.variable_scope("conv3D" + name):
		kernel = tf.get_variable("weights", shape=[length, height, width, input_dim, output_dim], \
			dtype=tf.float32, initializer=tf.constant_initializer(0.01))
		conv = tf.nn.conv3d(input_layer, kernel, stride, padding=padding)
	return conv

class NN_arch(object):
	def __init__(self):
		pass

	def build_graph(self, grid, activation=tf.nn.relu, is_training=True):
		self.layer1 = conv3DLayer(grid, 1, 8, 5, 5, 5, [1, 2, 2, 2, 1], name="layer1", activation=activation, is_training=is_training)
		self.layer2 = conv3DLayer(self.layer1, 8, 8, 5, 5, 5, [1, 2, 2, 2, 1], name="layer2", activation=activation, is_training=is_training)
		self.layer3 = conv3DLayer(self.layer2, 8, 8, 5, 5, 5, [1, 2, 2, 2, 1], name="layer3", activation=activation, is_training=is_training)
		self.layer4 = conv3DLayer(self.layer3, 8, 8, 5, 5, 5, [1, 1, 1, 1, 1], name="layer4", activation=activation, is_training=is_training)
		
		self.objectness = conv3D_to_output(self.layer4, 8, 2, 5, 5, 5, [1, 1, 1, 1, 1], name="objectness", activation=None)
		self.cordinate = conv3D_to_output(self.layer4, 8, 24, 5, 5, 5, [1, 1, 1, 1, 1], name="cordinate", activation=None)
		self.y = tf.nn.softmax(self.objectness, dim=-1)


	def build_graph1(self, grid, activation=tf.nn.relu, is_training=True):
		self.layer1 = conv3DLayer(grid, 1, 16, 5, 5, 5, [1, 2, 2, 2, 1], name="layer1", activation=activation, is_training=is_training)
		self.layer2 = conv3DLayer(self.layer1, 16, 32, 5, 5, 5, [1, 2, 2, 2, 1], name="layer2", activation=activation, is_training=is_training)
		self.layer3 = conv3DLayer(self.layer2, 32, 64, 3, 3, 3, [1, 2, 2, 2, 1], name="layer3", activation=activation, is_training=is_training)
		self.layer4 = conv3DLayer(self.layer3, 64, 64, 3, 3, 3, [1, 1, 1, 1, 1], name="layer4", activation=activation, is_training=is_training)
		
		self.objectness = conv3D_to_output(self.layer4, 64, 2, 3, 3, 3, [1, 1, 1, 1, 1], name="objectness", activation=None)
		self.cordinate = conv3D_to_output(self.layer4, 64, 24, 3, 3, 3, [1, 1, 1, 1, 1], name="cordinate", activation=None)
		self.y = tf.nn.softmax(self.objectness, dim=-1)
	

def grid_model(sess, grid_shape=(300, 300, 300),activation=tf.nn.relu, is_training=True):
	grid = tf.placeholder(tf.float32, [None, grid_shape[0], grid_shape[1], grid_shape[2], 1])
	phase_train = tf.placeholder(tf.bool, name='phase_train') if is_training else None
	with tf.variable_scope("3D_CNN_model") as scope:
		model_arch = NN_arch()
		model_arch.build_graph1(grid, activation=activation, is_training=phase_train)

	if is_training:
		initialized_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="3D_CNN_model")
		sess.run(tf.variables_initializer(initialized_var))
	return model_arch, grid, phase_train

def loss_function(model):
	g_map = tf.placeholder(tf.float32, model.cordinate.get_shape().as_list()[:4])
	g_cord = tf.placeholder(tf.float32, model.cordinate.get_shape().as_list())
	non_gmap = tf.subtract(tf.ones_like(g_map, dtype=tf.float32), g_map)

	elosion = 0.00001
	y = model.y
	is_obj_loss = -tf.reduce_sum(tf.multiply(g_map,  tf.log(y[:, :, :, :, 0] + elosion)))
	non_obj_loss = tf.multiply(-tf.reduce_sum(tf.multiply(non_gmap, tf.log(y[:, :, :, :, 1] + elosion))), 0.0008)
	cross_entropy = tf.add(is_obj_loss, non_obj_loss)
	obj_loss = cross_entropy

	g_cord = tf.placeholder(tf.float32, model.cordinate.get_shape().as_list())
	cord_diff = tf.multiply(g_map, tf.reduce_sum(tf.square(tf.subtract(model.cordinate, g_cord)), 4))
	cord_loss = tf.multiply(tf.reduce_sum(cord_diff), 0.02)
	return tf.add(obj_loss, cord_loss), obj_loss, cord_loss, is_obj_loss, non_obj_loss, g_map, g_cord, y

def create_optimizer(all_loss, lr=0.001):
	opt = tf.train.AdamOptimizer(lr)
	optimizer = opt.minimize(all_loss)
	return optimizer