from jax import random


class DataGenerator1D:
    """For 1D PDEs: (t, x) domain"""
    def __init__(self, domain=(0.0, 1.0, -1.0, 1.0), seed=0):
        self.t_min, self.t_max = domain[0], domain[1]
        self.x_min, self.x_max = domain[2], domain[3]
        self.key = random.PRNGKey(seed)

    def sample(self, batch_ic, batch_bc, batch_res, beta_b=1.0):
        self.key, k1, k2, k3, k4 = random.split(self.key, 5)

        x_ic = random.uniform(k1, (batch_ic,), minval=self.x_min, maxval=self.x_max)
        t_bc = random.uniform(k2, (batch_bc,), minval=self.t_min, maxval=self.t_max)

        t_raw = random.beta(k3, 1.0, beta_b, shape=(batch_res,))
        t_res = self.t_min + t_raw * (self.t_max - self.t_min)
        x_res = random.uniform(k4, (batch_res,), minval=self.x_min, maxval=self.x_max)

        return x_ic, t_bc, t_res, x_res


class DataGenerator2D:
    """For 2D PDEs: (t, x, y) domain"""
    def __init__(self, domain=(0.0, 10.0, 0.0, 5.0, -0.5, 0.5), seed=0):
        self.t_min, self.t_max = domain[0], domain[1]
        self.x_min, self.x_max = domain[2], domain[3]
        self.y_min, self.y_max = domain[4], domain[5]
        self.key = random.PRNGKey(seed)

    def sample(self, batch_ic, batch_bc, batch_res, beta_b=1.0):
        self.key, k1, k2, k3, k4, k5, k6, k7, k8 = random.split(self.key, 9)

        x_ic = random.uniform(k1, (batch_ic,), minval=self.x_min, maxval=self.x_max)
        y_ic = random.uniform(k2, (batch_ic,), minval=self.y_min, maxval=self.y_max)

        t_bc = random.uniform(k3, (batch_bc,), minval=self.t_min, maxval=self.t_max)
        x_bc = random.uniform(k4, (batch_bc,), minval=self.x_min, maxval=self.x_max)
        y_bc = random.uniform(k5, (batch_bc,), minval=self.y_min, maxval=self.y_max)

        t_raw = random.beta(k6, 1.0, beta_b, shape=(batch_res,))
        t_res = self.t_min + t_raw * (self.t_max - self.t_min)
        x_res = random.uniform(k7, (batch_res,), minval=self.x_min, maxval=self.x_max)
        y_res = random.uniform(k8, (batch_res,), minval=self.y_min, maxval=self.y_max)

        return x_ic, y_ic, t_bc, x_bc, y_bc, t_res, x_res, y_res

class DataGen_JFNK:
    def __init__(self, domain=(0.0, 1.0, -1.0, 1.0), seed=0):
        self.t_min, self.t_max = domain[0], domain[1]
        self.x_min, self.x_max = domain[2], domain[3]
        self.key = random.PRNGKey(seed)
    def sample(self, batch_ic, batch_bc, batch_res):
        self.key, k1, k2, k3, k4 = random.split(self.key, 5)
        x_ic  = random.uniform(k1, (batch_ic,),  minval=self.x_min, maxval=self.x_max)
        t_bc  = random.uniform(k2, (batch_bc,),  minval=self.t_min, maxval=self.t_max)
        t_res = random.uniform(k3, (batch_res,), minval=self.t_min, maxval=self.t_max)
        x_res = random.uniform(k4, (batch_res,), minval=self.x_min, maxval=self.x_max)
        return x_ic, t_bc, t_res, x_res