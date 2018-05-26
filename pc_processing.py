import numpy as np
import rospy
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2

def load_pc_from_pcd(pcd_path):
    """Load PointCloud data from pcd file."""
    p = pcl.load(pcd_path)
    return np.array(list(p), dtype=np.float32)

def load_pc_from_bin(bin_path):
    """Load PointCloud data from pcd file."""
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return obj

def proj_to_velo(calib_data):
    """Projection matrix to 3D axis for 3D Label"""
    rect = calib_data["R0_rect"].reshape(3, 3)
    velo_to_cam = calib_data["Tr_velo_to_cam"].reshape(3, 4)
    inv_rect = np.linalg.inv(rect)
    inv_velo_to_cam = np.linalg.pinv(velo_to_cam[:, :3])
    return np.dot(inv_velo_to_cam, inv_rect)


def filter_camera_angle(places):
    """Filter camera angles for KiTTI Datasets"""
    bool_in = np.logical_and((places[:, 1] < places[:, 0] - 0.27), (-places[:, 1] < places[:, 0] - 0.27))
    # bool_in = np.logical_and((places[:, 1] < places[:, 0]), (-places[:, 1] < places[:, 0]))
    return places[bool_in]

def publish_pc2(pc, obj):
    """Publisher of PointCloud data"""
    pub = rospy.Publisher("/points_raw", PointCloud2, queue_size=1000000)
    rospy.init_node("pc2_publisher")
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points = pc2.create_cloud_xyz32(header, pc[:, :3])

    pub2 = rospy.Publisher("/points_raw1", PointCloud2, queue_size=1000000)
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "velodyne"
    points2 = pc2.create_cloud_xyz32(header, obj)

    r = rospy.Rate(0.1)
    
    pub.publish(points)
    pub2.publish(points2)
    r.sleep()