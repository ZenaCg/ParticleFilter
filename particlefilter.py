import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import filtfilt
from scipy.signal import butter
from time import time
from skimage.io import imread, imsave
from utils import Grid, depth_from_kinect, color_i_from_kinect, color_j_from_kinect, Map

#%% Data loading

dataset = 20

with np.load("data/Encoders%d.npz"%dataset) as data:
    encoder_counts = data["counts"] # 4 x n encoder counts
    encoder_stamps = data["time_stamps"] # encoder time stamps

with np.load("data/Hokuyo%d.npz"%dataset) as data:
    lidar_angle_min = data["angle_min"] # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"] # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"] # angular distance between measurements [rad]
    lidar_range_min = data["range_min"] # minimum range value [m]
    lidar_range_max = data["range_max"] # maximum range value [m]
    lidar_ranges = data["ranges"]       # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load("data/Imu%d.npz"%dataset) as data:
    imu_angular_velocity = data["angular_velocity"] # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"] # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
  
with np.load("data/Kinect%d.npz"%dataset) as data:
    disp_stamps = data["disparity_time_stamps"] # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"] # acquisition times of the rgb images

t_start = max((np.min(encoder_stamps), np.min(lidar_stamps), np.min(imu_stamps), np.min(rgb_stamps), np.min(disp_stamps)))

encoder_stamps -= t_start
lidar_stamps -= t_start
imu_stamps -= t_start
disp_stamps -= t_start
rgb_stamps -= t_start

def rgb_img_no(t):
    return np.argmin(np.abs(t-rgb_stamps))

def disp_img_no(t):
    return np.argmin(np.abs(t-disp_stamps))

def get_rgbd(t):
    
    path_d = 'dataRGBD/Disparity{0}/disparity{0}_{1}.png'.format(dataset, disp_img_no(t+1))
    path_rgb = 'dataRGBD/RGB{0}/rgb{0}_{1}.png'.format(dataset, rgb_img_no(t+1))
    
    return imread(path_rgb), depth_from_kinect(imread(path_d))

#%% post-processing for lidar data

# current orientation o
# lidar readings z
def project_lidar(o, z):
    lidar_angle_increment_deg = lidar_angle_increment * 180.0 / np.pi
    angles = -1 * np.arange(-135.0, 135.0 + lidar_angle_increment_deg, lidar_angle_increment_deg) * np.pi / 180.0
    
    return np.vstack((np.cos(o), np.sin(o))) * 0.135 + np.vstack((np.cos(o + angles), np.sin(o + angles))) * z

r_mean = np.mean(lidar_ranges, axis=1)
r_max = np.max(lidar_ranges, axis=1)
r_min = np.min(lidar_ranges, axis=1)

legal = ~(np.abs(r_min - r_max) < 1.0)

lidar_angle_increment_deg = lidar_angle_increment * 180.0 / np.pi
angles = np.arange(-135.0, 135.0 + lidar_angle_increment_deg, lidar_angle_increment_deg) * np.pi / 180.0

ax=plt.subplot(111,projection='polar')
ax.plot(angles, lidar_ranges[:, 77])

#%%

lidar_projection_test = project_lidar(0.0, lidar_ranges[:, 77])
plt.plot(lidar_projection_test[0],lidar_projection_test[1])

#%% Signal preprocessing for IMU data

t_start = 0.0
t_stepize = 0.025
dT = t_stepize
t_end = min((np.max(encoder_stamps), np.max(lidar_stamps), np.max(imu_stamps)))

n_steps = int(t_end // t_stepize)

t_steps = np.arange(t_start, t_end, t_stepize)

b, a = butter(6, 10, fs=100)
imu_filtered = filtfilt(b, a, imu_angular_velocity[2])

encoder_interp = np.stack([np.diff(np.interp(t_steps, encoder_stamps, np.cumsum(encoder_counts[k]))) for k in range(4)])
imu_interp = np.interp(t_steps, imu_stamps, imu_filtered)

plt.plot(t_steps, imu_interp / np.pi * 180)
plt.plot(t_steps, (np.cumsum(imu_interp) * dT / np.pi * 180))
    
#%% Particle Filter

# Random noise functions for the velocity and the angle
rng = np.random.RandomState(seed=1234)
p_velocity_std = 0.15
p_velocity_noise = lambda : rng.normal(loc=0.0, scale=p_velocity_std, size=(n_particles, 1))

p_omega_std = 0.25
p_omega_noise = lambda : rng.normal(loc=0.0, scale=p_omega_std, size=(n_particles, 1))

# Perform prediction 
def p_predict(pos, ori, vel, om):
    om = p_omega_noise() + om
    vel = p_velocity_noise() + vel
    pos_pred = pos + np.concatenate((vel * dT * np.sinc(om * dT/2 / np.pi) * np.cos(ori + om * dT / 2),
                               vel * dT * np.sinc(om * dT/2 / np.pi) * np.sin(ori + om * dT / 2)), axis=1)
    
    ori_pred = ori + om * dT
    return pos_pred, ori_pred

# Lidar World to Robot transform
def lidar_w2r(pos, ori, angles, rang):
    x = pos[:, :1] + np.cos(ori) * 0.135 + np.cos(ori + angles.reshape(1, -1)) * rang.reshape(1, -1)
    y = pos[:, -1:] + np.sin(ori) * 0.135 + np.sin(ori + angles.reshape(1, -1)) * rang.reshape(1 ,-1)
    
    return np.stack((x, y))

# Update the weights with the new grid correlations
def update_weights(w, grid_corr):
    w_new = w * grid_corr
    return w_new / np.sum(w_new)

# Preduction step: sample from particles and perform prediction
def bootstrap(w, pos, ori, vel, om):
    
    new_p = np.searchsorted(np.cumsum(w), rng.uniform(size=w.shape))
    #new_p = np.arange(w.size)
    new_pos, new_ori = p_predict(pos[new_p], ori[new_p], vel, om)
    return w[new_p], new_pos, new_ori

#%% Set the grid map
    
# Initialise all data that we need
grid = Grid(0.15, [-25, 25], [-25, 25], np.array((0.0,)))
n_particles = 50
p_position = np.zeros(shape=(n_particles, 2, n_steps+1))
p_velocity = np.zeros(shape=n_steps+1)
p_orientation = np.zeros(shape=(n_particles, 1, n_steps+1))
p_accepted_position = np.zeros(shape=(2, n_steps))
p_accepted_orientation = np.zeros(shape=(1, n_steps))
p_weight = np.ones(shape=(n_particles, n_steps+1)) / float(n_particles)
#%%
lidar_angle_increment_deg = lidar_angle_increment * 180.0 / np.pi
angles = -1 * np.arange(-135.0, 135.0 + lidar_angle_increment_deg, 
                   lidar_angle_increment_deg) * np.pi / 180.0

# Initialise Grid with the first map that we have
grid.reset()
lidar_idx = np.argmin(np.abs(-dT-lidar_stamps))
valid = np.logical_and(legal, lidar_ranges[:, lidar_idx] > 0.1)
lidar_points = project_lidar(0.0, lidar_ranges[:, lidar_idx])[:, valid]
grid.update((0.0, 0.0), lidar_points, True)

#%% Iterations for the particle filter
    
for s in range(n_steps):
    t = dT * s
    # Look for the right lidar timestamp
    lidar_idx = np.argmin(np.abs(t-lidar_stamps))
    # Select only valid lidar data
    valid = np.logical_and(legal, lidar_ranges[:, lidar_idx] > 0.1)
    # Compute the lidar hits in the real world for each particle's position
    # Dimension: #Particles x #LiDAR points
    lidar_hits = lidar_w2r(p_position[:, :, s], p_orientation[:, :, s],
                           angles[valid], lidar_ranges[valid, lidar_idx])
    
    # Our observation probabilities go here...
    p_h = np.zeros(shape=(n_particles,))
    # Compute the grid correlation for each particles. Update the particle's
    # position according to where the best correlation occurred and store
    # the measurement probability p_h
    for p in range(n_particles):
        p_h[p], pos_x, pos_y = grid.correlate(lidar_hits[:, p])
        #print(p_h[p], p_position[p, :, s])
        p_position[p, :, s] += np.array((pos_x, pos_y))
    p_h -= np.max(p_h)
    # Update the weight using the old weight and the measurement probability
    # that we just computed
    p_weight[:, s] = update_weights(p_weight[:, s], np.exp(p_h))
    # We update the map selecting the point with the highest weighting
    update_sel = np.argmax(p_weight[:, s])
    update_pos = p_position[update_sel, :, s]
    p_accepted_position[:, s] = update_pos
    p_accepted_orientation[:, s] = p_orientation[update_sel, :, s]
    # Compute the LiDAR points for the selected position
    lidar_points = update_pos[:, None] + project_lidar(p_accepted_orientation[:, s], lidar_ranges[:, lidar_idx])[:, valid]
    # Perform the map update
    grid.update(update_pos, lidar_points, s % 3 == 0)
    
    # Update weight, position and orientation by predicting where we might be
    # in the next step
    omega = -imu_interp[s]
    p_velocity[s] = np.mean(encoder_interp[:, s]) * 0.0022 / dT
    p_weight[:, s+1], p_position[:, :, s+1], p_orientation[:, :, s+1] = bootstrap(p_weight[:, s], p_position[:, :, s], 
                                      p_orientation[:, :, s], p_velocity[s],
                                      omega)
    
    if s % 20 == 0:
        grid_exp = 1.0 - 1.0 / (1+np.exp(grid.grid))

        plt.close()
        plt.imshow(grid_exp.T, cmap='gray')
        u,v=grid.w2g(p_accepted_position[0][:s], p_accepted_position[1][:s])
        plt.plot(u, v, c='r')
        plt.title("Obstacle mapping + Path (Dataset {}, step {:d})".format(dataset, s))
        plt.tight_layout()
        plt.savefig("map_ds{}_step{:04d}.png".format(dataset, s))  
        
        print("It {:04d}".format(s))
        
#%%
grid_exp = 1.0 - 1.0 / (1+np.exp(grid.grid))

plt.close()
plt.imshow(grid_exp.T, cmap='gray')
u,v=grid.w2g(p_accepted_position[0], p_accepted_position[1])
plt.plot(u, v, c='r')
plt.title("Obstacle mapping + Path (Dataset {})".format(dataset))
plt.tight_layout()
plt.savefig("map_ds{}_step{}.png".format(dataset, s))   

#%% Precompute ground depth and coordinate mapping
mm = Map(0.1, (-20.0, 20.0), (-20.0, 20.0))

K = np.array(((585.05, 0.0, 242.9),
              (0.0, 585.05, 315.8),
              (0.0, 0.0, 1.0)))
K_inv = np.linalg.inv(K)

min_line = 150 # Ignore all other pixels above this
pitch = -0.35 #18.0 / 180 * np.pi
yaw = -0.021
y_cam = K_inv[0,0] * np.arange(480) + K_inv[0,2]
y_cam = y_cam.reshape(-1, 1)
x_cam = K_inv[1,1] * np.arange(0, 640) + K_inv[1,2]
x_cam = x_cam.reshape(1, -1)
o_camera = np.array((0.18, 0.005, 0.50))
theoretical_depth = o_camera[-1] / np.cos(np.pi/2.0 + pitch - np.arctan(y_cam)) 
theoretical_depth[:min_line] = 100

R_y = np.array(( (np.cos(pitch), 0.0, -np.sin(pitch)), 
                (0.0, 1.0, 0.0), 
                (np.sin(pitch), 0.0, np.cos(pitch))))
R_z = np.array(((np.cos(yaw), -np.sin(yaw), 0.0), 
                (np.sin(yaw), np.cos(yaw), 0.0), 
                (0.0, 0.0, 1.0)))
R_coord = np.array(((0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)))
t = o_camera
i_depth, j_depth = np.meshgrid(np.arange(480), np.arange(640))
i_depth = i_depth.T
j_depth = j_depth.T
pixels = np.stack((i_depth, j_depth, np.ones_like(i_depth)))

camera = (K_inv @ pixels.reshape(3, -1)).reshape(3, 480, 640)
camera_unproject = camera * theoretical_depth.reshape(1, 480, -1) / np.sqrt(np.sum(np.square(camera), axis=0, keepdims=True))
robot_coord = (R_coord @ camera_unproject.reshape(3, -1)).reshape(3, 480, 640)
robot_coord_rotate = ( R_y @ robot_coord.reshape(3, -1) ).reshape(3, 480, 640)
robot = (robot_coord_rotate + t.reshape(3, 1, 1))


#%%

for s in range(n_steps):
    t = dT * s
    rgb_img, d_img = get_rgbd(t)
    
    ground_points_mark = np.abs(d_img - theoretical_depth) < 0.1
    ground_points = np.argwhere(ground_points_mark)
    image_points_i = np.rint(color_i_from_kinect(ground_points[:, 0], d_img[ground_points_mark].reshape(-1))).astype(np.int)
    image_points_j = np.rint(color_j_from_kinect(ground_points[:, 1])).astype(np.int)
    valid_image_points = np.logical_and(np.logical_and(image_points_i >= 0, image_points_i < 480),
                                        np.logical_and(image_points_j >= 0, image_points_j < 640))
    image_points_i = image_points_i[valid_image_points]
    image_points_j = image_points_j[valid_image_points]
    
    colors = rgb_img[image_points_i, image_points_j]
    
    robot_ground_points = robot[:2, ground_points[valid_image_points, 0], ground_points[valid_image_points, 1]]
    
    mm.update(p_accepted_position[:, s:s+1], p_accepted_orientation[0, s], robot_ground_points[:,::10], colors[::10])
    if s % 100 == 0:
        print("It {}".format(s))

#%%
        
plt.imshow(mm.grid.transpose(1, 0, 2))
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False) 
plt.title("Dataset{}: Color Map".format(dataset))
plt.tight_layout()

#%%

ax = plt.subplot(131)
ax.imshow(rgb_img)
plt.title('RGB Image')

ax = plt.subplot(132)
ax.imshow(d_img)
plt.title('Disparity Image')

ax = plt.subplot(133)
ax.imshow(np.abs(d_img - theoretical_depth) < 0.1)
plt.title('Ground pixels (yellow)')

#%%

starting_idx = 150
plt.plot(np.arange(starting_idx, 480), theoretical_depth[starting_idx:], label='Theor. Depth')
plt.plot(np.arange(starting_idx, 480), d_img[starting_idx:, 320], label='Measured Depth')
plt.plot(np.arange(starting_idx, 480), theoretical_depth[starting_idx:, 0] - d_img[starting_idx:, 320], label="Diff")
plt.legend()
plt.title("Kinect h={:0.2f}m pitch={:0.0f}deg".format(o_camera[-1], pitch * 180 / np.pi))

#%%

from mpl_toolkits.mplot3d import Axes3D
ax=plt.subplot(111, projection='3d')
ax.scatter(robot_coord[0, 150:].reshape(-1)[::100], robot_coord[1, 150:].reshape(-1)[::100], robot_coord[2, 150:].reshape(-1)[::100])
ax.set_zlabel('z')
ax.set_ylabel('y')
ax.set_xlabel('x')
#%%

from mpl_toolkits.mplot3d import Axes3D
ax=plt.subplot(111, projection='3d')
ax.scatter(robot_coord_rotate[0, 150:].reshape(-1)[::100], robot_coord_rotate[1, 150:].reshape(-1)[::100], robot_coord_rotate[2, 150:].reshape(-1)[::100])
ax.set_zlabel('z')
ax.set_ylabel('y')
ax.set_xlabel('x')
#%%

from mpl_toolkits.mplot3d import Axes3D
ax=plt.subplot(111, projection='3d')
ax.scatter(robot[0, 150:].reshape(-1)[::100], robot[1, 150:].reshape(-1)[::100], robot[2, 150:].reshape(-1)[::100])
ax.set_zlabel('z')
ax.set_ylabel('y')
ax.set_xlabel('x')