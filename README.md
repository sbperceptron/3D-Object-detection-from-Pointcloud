# 3D-Object-detction-from-Pointcloud

# required packages
1. python 2.7
2. point cloud libraray
3. NVIDIA-CUDA package
4. ROS
5. tensorflow (virtual environment)

# Note:
The project is being developed and tested on ubuntu 14.04

# instructions to run the code:
# 1. open a new terminal and run,
roscore
# 2. open a new terminal
source ~/tensorflow/bin/activate
(assuming the virtual environment name of the tensorflow is "tensorflow")

cd <package_location>
(the location where the object detction package is located)
# before running the following change the pcd_path variable in test.py to the location of test data
python test.py

# 3. open a new terminal and run the publisher,
rosrun tf static_transform_publisher 0 0 0 0 0 0 1 map velodyne 10
# 4. open a new terminal and run rviz to visualise the test results,
rosrun rviz rviz


