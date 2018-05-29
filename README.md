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


# working:
1. single class object detction using 3 3D_CNN layers using the grid method from vote3deep


# instructions to run the code:
# 1. open a new terminal and run,
roscore
# 2. open a new terminal
source ~/tensorflow/bin/activate

(assuming the virtual environment name of the tensorflow is "tensorflow")

cd <package_location>

(the location where the object detection package is located)
# before running the following change the pcd_path variable in test.py to the location of test data
python test.py

# 3. open a new terminal and run the publisher,
rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map velodyne 10
# 4. open a new terminal and run rviz to visualise the test results,
rosrun rviz rviz

# references 
1. Vote3Deep:  Fast  Object  Detection  in  3D  Point  Clouds  Using  Efficient
Convolutional  Neural  Networks  https://arxiv.org/pdf/1609.06666.pdf
2. 3D  Fully  Convolutional  Network  for  Vehicle  Detection  in  Point  Cloud https://arxiv.org/pdf/1611.08069.pdf
3. https://github.com/yukitsuji/3D_CNN_tensorflow
