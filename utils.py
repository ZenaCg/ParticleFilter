import numpy as np

def bresenham2D(sx, sy, ex, ey):
  '''
  Bresenham's ray tracing algorithm in 2D.
  Inputs:
	  (sx, sy)	start point of ray
	  (ex, ey)	end point of ray
  '''
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return x, y


def mapCorrelation(im, x_im, y_im, vp, xs, ys):
  '''
  INPUT 
  im              the map 
  x_im,y_im       physical x,y positions of the grid map cells
  vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)  
  xs,ys           physical x,y,positions you want to evaluate "correlation" 

  OUTPUT 
  c               sum of the cell values of all the positions hit by range sensor
  '''
  nx = im.shape[0]
  ny = im.shape[1]
  xmin = x_im[0]
  xmax = x_im[-1]
  xresolution = (xmax-xmin)/(nx-1)
  ymin = y_im[0]
  ymax = y_im[-1]
  yresolution = (ymax-ymin)/(ny-1)
  nxs = xs.size
  nys = ys.size
  cpr = np.zeros((nxs, nys))
  for jy in range(0,nys):
    y1 = vp[1,:] + ys[jy] # 1 x 1076
    iy = np.int16(np.round((y1-ymin)/yresolution))
    for jx in range(0,nxs):
      x1 = vp[0,:] + xs[jx] # 1 x 1076
      ix = np.int16(np.round((x1-xmin)/xresolution))
      valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), \
			                        np.logical_and((ix >=0), (ix < nx)))
      cpr[jx,jy] = np.mean(im[ix[valid],iy[valid]])
  return cpr

class Grid(object):
    
    def __init__(self, delta, x_lim, y_lim, qry):
        self.grid_delta = delta
        self.grid_x = x_lim
        self.grid_y = y_lim
        self.qry = qry
        self.n_u = int((self.grid_x[1] - self.grid_x[0]) / self.grid_delta)
        self.n_v = int((self.grid_y[1] - self.grid_y[0]) / self.grid_delta)
        self.t_u = int(-self.grid_x[0] / self.grid_delta)
        self.t_v = int(-self.grid_y[0] / self.grid_delta)

        self.reset()

    def reset(self):
        self.grid = np.zeros(shape=(self.n_u, self.n_v), dtype=np.float)

    def w2g(self, x_world, y_world):
        u = np.clip(np.rint(x_world / self.grid_delta).astype(np.int) + self.t_u, 0, self.n_u-1)
        v = np.clip(np.rint(y_world / self.grid_delta).astype(np.int) + self.t_v, 0, self.n_v-1)
        return u, v
    
    def update(self, emit_pos, hit_pos, rasterise=False):
        if rasterise:
            for k in range(hit_pos.shape[1]):
                u_0, v_0 = self.w2g(emit_pos[0], emit_pos[1])
                u_l, v_l = self.w2g(hit_pos[0, k], hit_pos[1, k])
                u, v = bresenham2D(u_0, v_0, u_l, v_l)
                self.grid[u[:-1].astype(np.int), v[:-1].astype(np.int)] += np.log(1/4)
        u, v = self.w2g(hit_pos[0], hit_pos[1])
        self.grid[u, v] += np.log(4)
        self.grid = np.clip(self.grid, -10, 25)
    
    def correlate(self, hit_pos):
        x_coord = (np.arange(self.n_u) - self.t_u) * self.grid_delta
        y_coord = (np.arange(self.n_v) - self.t_v) * self.grid_delta
        corr = mapCorrelation(self.grid, x_coord, y_coord, hit_pos, self.qry, self.qry)
        x_max, y_max = np.unravel_index(np.argmax(corr), corr.shape)
        return np.max(corr), self.qry[x_max], self.qry[y_max]

def depth_from_kinect(arr):
    return 1.03 / (-0.00304 * arr + 3.31)

def color_i_from_kinect(depth_i, depth_arr):
    dd = 1.03 / depth_arr
    return (depth_i * 526.37 + dd * -4.5 * 1750.46 + 19276.0) / 585.051

def color_j_from_kinect(depth_j):
    return (depth_j * 526.37 + 16662.0) / 585.051

class Map(object):
    
    def __init__(self, delta, x_lim, y_lim):
        self.grid_delta = delta
        self.grid_x = x_lim
        self.grid_y = y_lim
        self.n_u = int((self.grid_x[1] - self.grid_x[0]) / self.grid_delta)
        self.n_v = int((self.grid_y[1] - self.grid_y[0]) / self.grid_delta)
        self.t_u = int(-self.grid_x[0] / self.grid_delta)
        self.t_v = int(-self.grid_y[0] / self.grid_delta)

        self.reset()

    def reset(self):
        self.grid = np.zeros(shape=(self.n_u, self.n_v, 3), dtype=np.float)
        self.cnt = np.zeros(shape=(self.n_u, self.n_v), dtype=np.float)

    def w2g(self, x_world, y_world):
        u = np.clip(np.rint(x_world / self.grid_delta).astype(np.int) + self.t_u, 0, self.n_u-1)
        v = np.clip(np.rint(y_world / self.grid_delta).astype(np.int) + self.t_v, 0, self.n_v-1)
        return u, v
    
    def update(self, position, orientation, robot_gnd_pnt, colors):
        R = np.array(((np.cos(orientation),-np.sin(orientation)), (np.sin(orientation), np.cos(orientation))))
        robot_wld_pnt = R @ robot_gnd_pnt + position
        u, v = self.w2g(robot_wld_pnt[0], robot_wld_pnt[1])
        self.grid[u, v] = colors / 255.0

