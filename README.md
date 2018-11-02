# 3D-Object-detction-from-Pointcloud-tf

# required packages
1. python 2.7
2. point cloud libraray
3. NVIDIA-CUDA package
4. ROS
5. tensorflow (virtual environment)


# Note:
The project is under development

Platform: ubuntu 14.04


# Description:
The architecture of the neural network essentially consists of 3 3D Sparse CNN (convolutional neural network) layers and one receptive field layer. The neural network is trained for detection on primarily three classes of objects namely persons, cars, and cyclist. KITTI vision benchmark suite velodyne point cloud dataset is used as the training and testing dataset for the 3D object detection. The dataset consists of thousands of point cloud frames which are 3D annotated with the object locations. The point cloud data from the dataset have on average 100k points per frame of the point cloud. The dataset constitutes a total of 7481 frames, 80 percent of the frames are used to train the network and 20 percent are used for testing.

# Achievements:
1. The hyper parameter tuning of the model is complete and it is observed that the model started to converge.

# Improvements:
1. Working on improving the model computational performance. Currently the model developed has multi processing enabled looking to include multi threading and CUDA processing
2. Multi class object detction using single model

# Hyper parameters:
Number of epochs: 100
Batch size =16
Learning rate=0.001
L2 weight decay=0.0001
momentum=0.9
optimizer= SGD

# Computational Performance
Average time/epoch: 5 hrs

# Models:
Cars:        SparseCNN-->Relu-->FullyConnected
Cyclists:    SparseCNN-->Relu-->SparseCNN-->Relu-->FullyConnected
pedestrians: SparseCNN-->Relu-->SparseCNN-->Relu-->SparseCNN-->Relu-->FullyConnected

# references 
1. Vote3Deep:  Fast  Object  Detection  in  3D  Point  Clouds  Using  Efficient
Convolutional  Neural  Networks  https://arxiv.org/pdf/1609.06666.pdf
2. 3D  Fully  Convolutional  Network  for  Vehicle  Detection  in  Point  Cloud https://arxiv.org/pdf/1611.08069.pdf
3. https://github.com/yukitsuji/3D_CNN_tensorflow
