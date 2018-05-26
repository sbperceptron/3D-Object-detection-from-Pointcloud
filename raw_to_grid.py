import numpy as np

def raw_to_grid(pc, resolution=0.50, x=(0, 90), y=(-50, 50), z=(-4.5, 5.5), i=(0)):
    
    #extremes already known x, y, z, i indicate the extremes
    """Convert PointCloud2 to GRID"""
    logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
    logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
    logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])
    logic_i = np.greater_equal(pc[:, 3] , i)
    pointcloud = pc [np.logical_and(np.logical_and(logic_x, np.logical_and(logic_y, logic_z)),logic_i)]
    intensity=pointcloud[:,3]
    pointcloud =((pointcloud[:,:3] - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
    pointcloud=(np.array([(pointcloud[:,0]).astype(np.int32),pointcloud[:,1],pointcloud[:,2],intensity],dtype=np.float32)).T
    pointcloud=pointcloud [np.lexsort((pointcloud[:,3],pointcloud[:,2],pointcloud[:,1],pointcloud[:,0]))]
    grid_list,index,inverse,counts=np.unique(pointcloud[:,:3],axis=0,return_index=True,return_inverse=True,return_counts=True)
    grid_list=grid_list.astype(np.int32)
    i_mean=[]
    
    for i in range(len(index)-1):
        sum=0
        range1=(index[i],index[i+1]-1)
        sum=np.sum(pointcloud[:,3][range1[0]:range1[1]])
        i_mean.append(sum)
    i_mean.append(pointcloud[:,3][len(index)])
    grid = np.zeros((int((x[1] - x[0]) / resolution), int((y[1] - y[0]) / resolution), int(round((z[1]-z[0]) / resolution))))
    grid[grid_list[:, 0], grid_list[:, 1], grid_list[:, 2]] = i_mean

#     logic_x = np.logical_and(pc[:, 0] >= x[0], pc[:, 0] < x[1])
#     logic_y = np.logical_and(pc[:, 1] >= y[0], pc[:, 1] < y[1])
#     logic_z = np.logical_and(pc[:, 2] >= z[0], pc[:, 2] < z[1])
#     pc = pc[:, :3][np.logical_and(logic_x, np.logical_and(logic_y, logic_z))]
#     pc =((pc - np.array([x[0], y[0], z[0]])) / resolution).astype(np.int32)
#     grid = np.zeros((int((x[1] - x[0]) / resolution), int((y[1] - y[0]) / resolution), int(round((z[1]-z[0]) / resolution))))
#     grid[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    
    return grid


    
