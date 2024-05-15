import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


class NominalControlPD:
    def __init__(self, **kwargs):
        self.target = kwargs.get("target")
        self.umin = kwargs.get("umin")
        self.umax = kwargs.get("umax")
        Kp = 1
        Kw = 2

        self.nominal_policy = lambda x,t: np.clip([[Kw*np.arctan2(np.cos(x[2])*-(x[0]-self.target[0])+np.sin(x[2])*-(x[1]-self.target[1]),-np.sin(x[2])*-(x[0]-self.target[0])+np.cos(x[2])*-(x[1]-self.target[1])),Kp*(np.linalg.norm(self.target[0:2]-x[0:2]))]], self.umin, self.umax)
        # self.nominal_policy = lambda x,t: np.clip([[Kw*np.arctan2(np.cos(x[2])*(x[0]-self.target[0])+np.sin(x[2])*(x[1]-self.target[1]),-np.sin(x[2])*(x[0]-self.target[0])+np.cos(x[2])*(x[1]-self.target[1])),Kp*(np.linalg.norm(self.target[0:2]-x[0:2]))]], self.umin, self.umax)
    
    def get_nominal_control(self, x, t):
        # print(np.arctan2(np.cos(x[2])*(x[0]-self.target[0])+np.sin(x[2])*(x[1]-self.target[1]),-np.sin(x[2])*(x[0]-self.target[0])+np.cos(x[2])*(x[1]-self.target[1])))

        return self.nominal_policy(x,t)

    def get_nominal_controller(self, target):
        return self.nominal_policy


class NominalPolicy:

    def __init__(self, ctrl):
        self.ctrl = ctrl

    def __call__(self, x, t):
        return self.ctrl.get_nominal_control(x, t)

    def save_measurements(self, state, control, time):
        return {"dist_to_goal": np.linalg.norm(state[..., :2] - self.ctrl.target[:2], axis=-1)}
