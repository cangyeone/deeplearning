import os 
import multiprocessing 
import numpy as np 
import pykonal# 计算走时程序，需要进行安装
import cv2 
import pickle 
from pyproj import Proj 
from ttcrpy.rgrid import Grid2d# 可以同时进行射线追踪
import tqdm 
from scipy.interpolate import griddata 
N = 256     
def ray_tracing_fmm(src, rcv, slowness):
    #N = 1024        # 网格大小
    dx = 1200/N        # 网格宽度（km）
    # coordinates of the nodes
    xn = np.arange(0, (N+1)*dx, dx) - N // 2 * dx 
    yn = np.arange(0, (N+1)*dx, dx) - N // 2 * dx
    solver_c = pykonal.EikonalSolver(coord_sys='cartesian')
    solver_c.velocity.min_coords = xn[0], yn[0], 0
    solver_c.velocity.node_intervals = dx, dx, 1
    solver_c.velocity.npts = N, N, 1
    solver_c.velocity.values = 1/slowness[:, :, np.newaxis] # Uniform velocity model
    src_idx = (int((src[0, 0]-xn[0])/dx), int((src[0, 1]-yn[0])/dx), 0) 
    solver_c.traveltime.values[src_idx] = 0
    solver_c.unknown[src_idx] = False
    solver_c.trial.push(*src_idx)
    solver_c.solve()
    ttt2 = []
    for r in rcv:
        rx, ry, rz = int((r[0]-xn[0])/dx), int((r[1]-yn[0])/dx), 0
        tt = solver_c.traveltime.values[rx, ry, rz] 
        ttt2.append(tt) 
    xi = ((rcv[:, 0] - xn[0])/dx).astype(np.int32) 
    yi = ((rcv[:, 1] - yn[0])/dx).astype(np.int32) 
    ttt2 = np.array(ttt2) 
    src_idx = (int((src[0, 0]-xn[0])/dx), int((src[0, 1]-yn[0])/dx))
    xn2, yn2 = np.meshgrid(xn[:-1], yn[:-1]) 
    tt2d = griddata(rcv[:, :2], ttt2, (xn2, yn2), method="linear", fill_value=0)
    return ttt2, (xi, yi), tt2d 
def ray_tracing(src, rcv, slowness):
    #N = 1024        # 网格大小
    dx = 1200/N        # 网格宽度（km）
    # coordinates of the nodes
    xn = np.arange(0, (N+1)*dx, dx) - N // 2 * dx 
    yn = np.arange(0, (N+1)*dx, dx) - N // 2 * dx
    # create grid with default values (by default, slowness is defined for cells)
    grid = Grid2d(xn, yn)
    #print(slowness.shape)
    grid.set_slowness(slowness)
    ttt = grid.raytrace(src, rcv)
    xi = ((rcv[:, 0] - xn[0])/dx).astype(np.int32) 
    yi = ((rcv[:, 1] - yn[0])/dx).astype(np.int32) 
    src_idx = (int((src[0, 0]-xn[0])/dx), int((src[0, 1]-yn[0])/dx))
    xn2, yn2 = np.meshgrid(xn[:-1], yn[:-1]) 
    tt2d = griddata(rcv[:, :2], ttt, (xn2, yn2), method="linear")
    return ttt, tt2d 
def get_proj():
    if os.path.exists("ckpt/proj.pkl"):
        with open("ckpt/proj.pkl", "rb") as f:
            proj = pickle.load(f)
    else:
        file_ = open("tomodata/data/all.12", "r", encoding="utf-8")
        data = []
        for line in file_.readlines():
            sline = [i for i in line.strip().split(" ") if len(i)>0]
            p1 = (float(sline[1]), float(sline[2]))     
            p2 = (float(sline[3]), float(sline[4])) 
            data.append(p1)
            data.append(p2) 
        data = np.array(data) 
        lb = np.min(data, axis=0)
        ub = np.max(data, axis=0)
        cx = (lb+ub)/2
        proj = Proj(f"+proj=sterea +lon_0={cx[1]} +lat_0={cx[0]} +units=km")
        with open("ckpt/proj.pkl", "wb") as f:
            pickle.dump(proj, f)
    return proj 

class DataTrain():
    def __init__(self):
        base_dir = "train2014"
        self.file_names = [os.path.join(base_dir, n) for n in os.listdir(base_dir)]
        self.fqueue = multiprocessing.Queue(100)
        self.dqueue = multiprocessing.Queue(100)
        multiprocessing.Process(target=self.feed, args=(self.fqueue, )).start() 
        self.count = 0 
        for n in range(32):
            multiprocessing.Process(target=self.sample, args=(self.fqueue, self.dqueue)).start()
    def next_batch(self, batch_size=32):
        X = [] 
        D = []
        T = []
        S = []
        for b in range(batch_size):
            V, T = self.dqueue.get() 
            #print(V.shape, T.shape)
            X.append(T) 
            D.append(V)
        X = np.concatenate(X, axis=0) 
        D = np.concatenate(D, axis=0)
        return X, D
    def feed(self, fqueue):
        while True:
            for n in self.file_names:
                if os.path.exists(n)==False:continue 
                img = cv2.imread(n) 
                if type(img)!=np.ndarray:
                    continue 
                if len(img)==0:continue 
                fqueue.put(img)
    def sample(self, fqueue, dqueue): 
        file_ = open("tomodata/data/all.12", "r", encoding="utf-8")
        station = {} 
        proj = get_proj()
        for line in file_.readlines():
            sline = [i for i in line.strip().split(" ") if len(i)>0] 
            st1 = sline[-2]
            st2 = sline[-1] 
            y1a, x1a = float(sline[1]), float(sline[2]) 
            y2a, x2a = float(sline[3]), float(sline[4]) 
            x1, y1 = proj(x1a, y1a)
            x2, y2 = proj(x2a, y2a)
            if st1 not in station:
                station[st1] = [x1, y1] 
            if st2 not in station:
                station[st2] = [x2, y2]
        file_.close()
        location = [] 
        for key in station:
            loc = station[key] 
            location.append(loc)
        location = np.array(location)
        NL = len(location)
        #print(location.shape)
        while True:
            img = fqueue.get() 
            h, w, c = img.shape 
            K = NL 
            KK = np.random.randint(int(K*0.9), K)
            location = np.random.uniform(-500, 500, [KK, 2])
            try:
                img = cv2.resize(img, [300, 300], interpolation=cv2.INTER_LINEAR)
                c = np.random.randint(25, 50)
                w = np.random.randint(150)
                img = img[c:c+w, c:c+w, :]
                img = cv2.resize(img, [N, N], interpolation=cv2.INTER_LINEAR) 
                img = cv2.GaussianBlur(img, [51, 51], sigmaX=10)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                #img = cv2.resize(img, [N, N], interpolation=cv2.INTER_LINEAR)
                img = img.astype(np.float32) 
                img = np.mean(img, axis=2)
                velo = img / 255 * 2.5 + 2.5  
            except:
                velo = np.ones([N, N])
            #print(velo.shape)
            loc = np.copy(location)
            idx = np.arange(len(loc))
            np.random.shuffle(idx)
            T = []
            num = 100#np.random.randint(100)
            for n in idx[:num]:
                #if np.random.random()<0.8:
                #    continue 
                src = loc[n:n+1] 
                rcv = loc[:] 
                #print(velo.shape)
                ttt, _, tt2d = ray_tracing_fmm(src, rcv, 1/velo)
                arrival_time = np.zeros([N, N, 3])
                arrival_time[:, :-1, 0] = np.clip((tt2d[:, 1:]-tt2d[:, :-1]), -10, 10)
                arrival_time[:-1, :, 1] = np.clip((tt2d[1:, :]-tt2d[:-1, :]), -10, 10)
                arrival_time[:, :, 2] = tt2d / 100 
                T.append(arrival_time[:, :])
            T = np.stack(T, axis=0)
            dqueue.put([velo[np.newaxis, ..., np.newaxis], T[np.newaxis, ...]]) 