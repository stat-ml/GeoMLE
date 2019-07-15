__all__ = ('DataGenerator')

import pandas as pd
import numpy as np
from .utils import bound_nonuniform_sampler, uniform_sampler

class DataGenerator():

    def __init__(self,
                 random_state: int = None,
                 type_noise:   str = 'norm'):

        self.set_rng(random_state)
        self.set_gen_noise(type_noise)
        self.dict_gen = {
            #syntetic data
            'Helix1d':        gen_helix1_data,
            'Helix2d':        gen_helix2_data,
            'Helicoid':       gen_helicoid_data,
            'Spiral':         gen_spiral_data,
            'Roll':           gen_roll_data,
            'Scurve':         gen_scurve_data,
            'Star':           gen_star_data,
            'Moebius':        gen_moebius_data,

            'Sphere':         gen_sphere_data,
            'Norm':           gen_norm_data,
            'Uniform':        gen_uniform_data,
            'Cubic':          gen_cubic_data,

            'Affine_3to5':    gen_affine3_5_data,
            'Affine':         gen_affine_data,

            'Nonlinear_4to6': gen_nonlinear4_6_data,
            'Nonlinear':      gen_nonlinear_data,
            'Paraboloid':     gen_porabaloid_data,

            # #real data
            # 'Digits':         get_digits,
            # 'Isomap':         get_Isomap,
            # 'Hands':          get_Hands,
            # 'ISOLET':         get_ISOLET,
            # 'MNISTd':         get_MNISTd
        }


    def set_rng(self, random_state:int=None):
        if random_state is not None:
            np.random.seed(random_state)

    def set_gen_noise(self, type_noise:str):   
        if not hasattr(self, 'rng'):
            self.set_rng()
        if type_noise == 'norm':
            self.gen_noise = np.random.randn
        if type_noise == 'uniform':
            self.gen_noise = lambda n, dim: np.random.rand(n, dim) - 0.5

    def gen_data(self, name:str, n:int, dim:int, d:int, type_sample:str='uniform', noise:float=0.0):
    # Parameters:
    # --------------------
    # name: string 
    #     Type of generetic data
    # n: int
    #     The number of sample points
    # dim: int
    #     The dimension of point
    # d: int
    #     The hyperplane dimension
    # noise: float, optional(default=0.0)
    #     The value of noise in data 

    # Returns:
    # data: pd.Dataframe of shape (n, dim)
    #     The points
        assert name in self.dict_gen.keys(),\
               'Name of data is unknown'
        if (type_sample == 'uniform'):
            if name == 'Sphere':
                sampler = np.random.randn
            else:
                sampler = np.random.rand
        elif (type_sample == 'nonuniform'):
            if name == 'Sphere':
                sampler = uniform_sampler
            else:
                sampler = bound_nonuniform_sampler
        else:
            assert False, 'Check type_sample'
        
        data = self.dict_gen[name](n=n, dim=dim, d=d, sampler=sampler)
        noise = self.gen_noise(n, dim) * noise
        
        return  data + noise
               

#############################################################################
#                                SYNTETIC DATA                              #
#############################################################################
from sklearn import datasets as ds

def gen_spiral_data(n, dim, d, sampler):
    assert d < dim
    assert d == 1
    assert dim >= 3
    t = 10 * np.pi * sampler(n)
    data = pd.DataFrame(np.vstack([100 * np.cos(t), 100 * np.sin(t),
                                   t, np.zeros((dim - 3, n))])).T
    assert data.shape == (n, dim)
    return data

def gen_helix1_data(n, dim, d, sampler):
    assert d < dim
    assert d == 1
    assert dim >= 3
    t = 2 * np.pi / n + sampler(n) * 2 * np.pi
    data = pd.DataFrame(np.vstack([(2 + np.cos(8*t))*np.cos(t),
                                   (2 + np.cos(8*t))*np.sin(t),
                                   np.sin(8*t), np.zeros((dim - 3, n))])).T
    assert data.shape == (n, dim)
    return data

def gen_helix2_data(n, dim, d, sampler):
    assert d < dim
    assert d == 2
    assert dim >= 3
    r = 10 * np.pi * sampler(n)
    p = 10 * np.pi * sampler(n)
    data = pd.DataFrame(np.vstack([r*np.cos(p), r*np.sin(p),
                                   0.5*p, np.zeros((dim - 3, n))])).T
    assert data.shape == (n, dim)
    return data

def gen_helicoid_data(n, dim, d, sampler):
    assert d <= dim
    assert d == 2
    assert dim >= 3
    u = 2 * np.pi / n + sampler(n) * 2 * np.pi
    v = 5 * np.pi * sampler(n)
    data = pd.DataFrame(np.vstack([np.cos(v),
                                   np.sin(v) * np.cos(v),
                                   u,
                                   np.zeros((dim - 3, n))])).T
    assert data.shape == (n, dim)
    return data
    
def gen_roll_data(n, dim, d, sampler):
    assert d < dim
    assert dim >= 3
    assert d == 2
    t = 1.5 * np.pi * (1 + 2 * sampler(n))
    p = 21 * sampler(n)
    
    data = pd.DataFrame(np.vstack([t * np.cos(t),
                                   p,
                                   t * np.sin(t),
                                   np.zeros((dim - d - 1, n))])).T
    assert data.shape == (n, dim)
    return data

def gen_scurve_data(n, dim, d, sampler):
    assert d < dim
    assert dim >= 3
    assert d == 2
    t = 3 * np.pi * (sampler(n) - 0.5)
    p = 2.0 * sampler(n)
    
    data = pd.DataFrame(np.vstack([np.sin(t),
                                   p,
                                   np.sign(t) * (np.cos(t) - 1),
                                   np.zeros((dim - d - 1, n))])).T
    assert data.shape == (n, dim)
    return data


def gen_sphere_data(n, dim, d, sampler):
    assert d < dim
#     V = np.random.randn(n, d + 1)
    V = sampler(n, d+1)
    data = pd.DataFrame(np.hstack([V/np.sqrt((V**2).sum(axis=1))[:,None],
                                   np.zeros((n, dim - d - 1))]))
    assert data.shape == (n, dim)
    return data
    
def gen_norm_data(n, dim, d, sampler):
    assert d <= dim
    norm_xyz = np.random.multivariate_normal(np.zeros(d), np.identity(d), n)
    data = pd.DataFrame(np.hstack([norm_xyz, np.zeros((n, dim - d))]))
    assert data.shape == (n, dim)
    return data
    
def gen_uniform_data(n, dim, d, sampler):
    assert d <= dim
    uniform_xyz = np.random.uniform(size=(n, d))
    data = pd.DataFrame(np.hstack([uniform_xyz, np.zeros((n, dim - d))]))
    assert data.shape == (n, dim)
    return data

def gen_cubic_data(n, dim, d, sampler):
    assert d < dim
    cubic_data = np.array([[]]*(d + 1))
    for i in range(d + 1):
        n_once = int(n / (2 * (d + 1)) + 1)
        #1st side
        data_once = sampler(d + 1, n_once)
        data_once[i] = 0
        cubic_data = np.hstack([cubic_data, data_once])
        #2nd side
        data_once = sampler(d + 1, n_once)
        data_once[i] = 1
        cubic_data = np.hstack([cubic_data, data_once])
    cubic_data = cubic_data.T[:n]
    data = pd.DataFrame(np.hstack([cubic_data, np.zeros((n, dim - d - 1))]))
    assert data.shape == (n, dim)   
    return data
    
def gen_moebius_data(n, dim, d, sampler):
    assert dim == 3
    assert d == 2
    
    phi = sampler(n) * 2 * np.pi
    rad = sampler(n) * 2 - 1
    data = pd.DataFrame(np.vstack([(1+0.5*rad*np.cos(5.0*phi))*np.cos(phi),
                                   (1+0.5*rad*np.cos(5.0*phi))*np.sin(phi),
                                   0.5*rad*np.sin(5.0*phi)])).T

    assert data.shape == (n, dim)
    return data

def gen_affine_data(n, dim, d, sampler):
    assert dim >= d

    p = sampler(d, n) * 5 - 2.5
    v = np.eye(dim, d)
#     v = np.random.randint(0, 10, (dim, d))
    data = pd.DataFrame(v.dot(p).T)
    
    assert data.shape == (n, dim)
    return data    

def gen_affine3_5_data(n, dim, d, sampler):
    assert dim == 5
    assert d == 3

    p = 4 * sampler(d, n)
    A = np.array([[ 1.2, -0.5, 0],
                  [ 0.5,  0.9, 0],
                  [-0.5, -0.2, 1],
                  [ 0.4, -0.9, -0.1],
                  [ 1.1, -0.3, 0]])
    b = np.array([[3, -1, 0, 0, 8]]).T
    data = A.dot(p) + b
    data = pd.DataFrame(data.T)

    assert data.shape == (n, dim)
    return data

def gen_nonlinear4_6_data(n, dim, d, sampler):
    assert dim == 6
    assert d == 4

    p0, p1, p2, p3 = sampler(d, n)
    data = pd.DataFrame(np.vstack([p1**2 * np.cos(2*np.pi*p0),
                                   p2**2 * np.sin(2*np.pi*p0),
                                   p1 + p2 + (p1-p3)**2,
                                   p1 - 2*p2 + (p0-p3)**2,
                                  -p1 - 2*p2 + (p2-p3)**2,
                                   p0**2 - p1**2 + p2**2 - p3**2])).T

    assert data.shape == (n, dim)
    return data

def gen_nonlinear_data(n, dim, d, sampler):
    assert dim >= d
    m = int(dim / (2 * d))
    assert dim == 2 * m * d

    p = sampler(d, n)
    F = np.zeros((2*d, n))
    F[0::2, :] = np.cos(2*np.pi*p)
    F[1::2, :] = np.sin(2*np.pi*p)
    R = np.zeros((2*d, n))
    R[0::2, :] = np.vstack([p[1:], p[0]])
    R[1::2, :] = np.vstack([p[1:], p[0]])
    D = (R * F).T
    data = pd.DataFrame(np.hstack([D] * m))

    assert data.shape == (n, dim)
    return data
    
def gen_porabaloid_data(n, dim, d, sampler):
    assert dim == 3 * (d + 1)

    E = np.random.exponential(1, (d+1, n))
    X = ((1 + E[1:]/E[0])**-1).T
    X = np.hstack([X, (X ** 2).sum(axis=1)[:,np.newaxis]])
    data = pd.DataFrame(np.hstack([X, np.sin(X), X**2]))

    assert data.shape == (n, dim)
    return data

def gen_star_data(n, dim, d, sampler):
    assert dim >= d
    assert d == 1
    assert dim >= 2

    t = np.pi - sampler(n) * 2 * np.pi
    omega = 5
    X = np.concatenate((((1 + 0.3*np.cos(omega*t))*np.cos(t)).reshape(-1, 1),
                        ((1 + 0.3*np.cos(omega*t))*np.sin(t)).reshape(-1, 1),
                        np.zeros((n, dim - 2))), axis=1)

    data = pd.DataFrame(X)
    assert data.shape == (n, dim)
    return data 

#############################################################################
#                                  REAL DATA                                #
#############################################################################
# from scipy.io import loadmat
# import zipfile
# from PIL import Image
# import io
# from os.path import dirname, join

# def get_digits(n=1797, dim=64, d=10):
#     assert (n, dim, d) == (1797, 64, 10)
    
#     data = ds.load_digits()
#     data = pd.DataFrame(data['data'])
    
#     assert data.shape == (n, dim)
#     return data


# def get_Isomap(n=698, dim=4096, d=3):
#     assert (n, dim, d) == (698, 4096, 3)
    
#     module_path = dirname(__file__)
#     path = join(module_path, 'data', 'isomap', 'face_data.mat')
#     mat = loadmat(path)
#     data = pd.DataFrame(mat['images']).T
    
#     assert data.shape == (n, dim)
#     return data

# def get_Hands(n=481, dim=245760, d=3):
#     assert (n, dim, d) == (481, 245760, 3)
    
#     module_path = dirname(__file__)
#     path = join(module_path, 'data', 'hands', 'hands.zip')
#     archive = zipfile.ZipFile(path, 'r')
#     data = []
#     for file in archive.filelist:
#         data_tmp = archive.read(file)
#         img = Image.open(io.BytesIO(data_tmp))
#         data.append(np.array(img).reshape(-1))
#     data = pd.DataFrame(np.array(data))
    
#     assert data.shape == (n, dim)
#     return data

# def loadMNIST(prefix, folder ):
#     intType = np.dtype('int32' ).newbyteorder( '>' )
#     nMetaDataBytes = 4 * intType.itemsize

#     data = np.fromfile(folder + "/" + prefix + '-images.idx3-ubyte',
#                         dtype = 'ubyte' )
#     magicBytes, nImages,\
#     width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
#     data = data[nMetaDataBytes:].astype(dtype = 'float32')
#     data = data.reshape([nImages, width, height])

#     labels = np.fromfile( folder + '/' + prefix + '-labels.idx1-ubyte',
#                           dtype = 'ubyte' )[2 * intType.itemsize:]
#     return data, labels

# def get_MNISTd(n=70000, dim=784, d = 0):
#     assert dim == 784
#     assert (n, d) == (6903, 0) or (n, d) == (7877, 1) or \
#            (n, d) == (6990, 2) or (n, d) == (7141, 3) or \
#            (n, d) == (6824, 4) or (n, d) == (6313, 5) or \
#            (n, d) == (6876, 6) or (n, d) == (7293, 7) or \
#            (n, d) == (6825, 8) or (n, d) == (6958, 9) or \
#            (n, d) == (70000, 10)
#     assert (d >= 0) and (d <= 10)
    
#     module_path = dirname(__file__)
#     path = join(module_path, 'data', 'mnist')
#     trainingImages, trainingLabels = loadMNIST('train', path)
#     testImages, testLabels = loadMNIST('t10k', path)
#     data = np.vstack([trainingImages, testImages]).reshape(70000, -1)
#     data = pd.DataFrame(data)
#     label = np.concatenate([trainingLabels, testLabels])
#     if d != 10:
#         mask = label == d
#         data = data.loc[mask]
    
#     assert data.shape[1] == dim
#     return data

# def get_ISOLET(n=7797, dim=617, d=19):
#     assert (n, dim) == (7797, 617)
#     assert (d >= 16) and (d <= 22)
    
#     module_path = dirname(__file__)
#     path = join(module_path, 'data', 'isolet', 'isolet_csv')
#     df = pd.read_csv(path)
#     data = df[[col for col in df.columns if 'f' in col]]

#     assert data.shape == (n, dim)
#     return data