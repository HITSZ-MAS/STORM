import math
import numpy as np
import scipy.linalg as linalg
from cvxopt import matrix, solvers
import sympy as sp

from BSpline import BSpline
        
class Knots_Optimizer_base:
    def __init__(self, CPs, t_max, t_min, V_max, A_max, J_max, p, constraints_on_every_knot=False, decay_factor=0.2):
        self.CPs = CPs
        self.t_max = t_max
        self.t_min = 0
        self.V_max = V_max
        self.A_max = A_max
        self.J_max = J_max
        self.n = CPs.shape[0]
        self.p = p
        self.t = np.array([t_max/(self.n-p)]*(self.n-p))
        self.BSpline = BSpline(self.get_t(), p)
        self.constraints_on_every_knot = constraints_on_every_knot
        self.base_decay_factor = decay_factor
        self.decay_factor = decay_factor * np.ones(self.t.shape[0])
        self.decay_history = []

    def update_decay_factor(self, epoch):
        self.decay_factor = self.base_decay_factor / math.sqrt(epoch)

    @staticmethod
    def compute_min_distance(CPs):
        '''
        Parameters:
            CPs: n * 3 numpy array
        Returns:
            t_min: float
        '''
        n = CPs.shape[0]
        d_min = np.inf
        for i in range(n):
            for j in range(i+1, n):
                _d = np.linalg.norm(CPs[i] - CPs[j])
                if _d < d_min:
                    d_min = _d
        return d_min
    
    def update(self, t, CPs):
        self.t = t
        self.CPs = CPs

    def update_t(self, t):
        self.t = t
        self.BSpline = BSpline(self.get_t(), self.p)
    
    def get_t(self):
        t = self.t
        t = [sum(self.t[:i]) for i in range(self.t.shape[0]+1)]
        t = t[0:1] * self.p + t + t[-1:] * self.p

        return t
    
    def update_tmax(self, t):
        self.t_max = t[-1]

    
    def update_CPs(self, CPs):
        self.CPs = CPs

    def get_c(self):
        c = np.ones((self.t.shape[0], 1))
        return c
    
    def _get_G_h(self):
        n = self.t.shape[0]
        _G_1 = -np.eye(n)
        _G_2 = np.ones((2, n))
        _G_2[1, :] = -1
        _G_3 = -np.eye(n)

        G = np.vstack((_G_1, _G_2, _G_3))
        
        h_1 = np.zeros((n, 1))
        h_2 = np.ones((2, 1))
        h_3 = (self.decay_factor - 1) * self.t
        h_3 = h_3.reshape(-1, 1)

        h_2[0] = self.t_max
        # h_2[1] = -self.t_min
        h_2[1] = 0
        h = np.vstack((h_1, h_2, h_3))

        if self.constraints_on_every_knot:
            _G = -np.eye(n)
            _h = -np.zeros((n, 1)) 
            G = np.vstack((G, _G))
            h = np.vstack((h, _h))

        return G, h
    
    def get_G_h(self):
        raise NotImplementedError("Subclass must implement get_G_h method")
    
    def optimize_knots(self):
        G, h = self.get_G_h()

        G = matrix(G, (G.shape[0], G.shape[1]), 'd')
        h = matrix(h, (h.shape[0], h.shape[1]), 'd')
        c = matrix(self.get_c(), (self.get_c().shape[0], self.get_c().shape[1]), 'd')
        init_guess = {'x': matrix(self.t.reshape(-1, 1), (self.t.shape[0], 1), 'd')}
        sol = solvers.lp(c, G, h)

        try:
            solvers.options['show_progress'] = False
            sol = solvers.lp(c, G, h, initvals=init_guess)
            if sol['status'] != 'optimal':
                raise RuntimeError(f"Optimization failed, status: {sol['status']}")
            optimal_knots = np.array(sol['x']).reshape(-1)
            decay = (self.t - optimal_knots) / self.t
            self.decay_history.append(decay)
            self.t = optimal_knots
            return optimal_knots
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {str(e)}")

class Knots_Optimizer_base_Simple(Knots_Optimizer_base):
    def __init__(self, CPs, t_max, t_min, V_max, A_max, J_max, p, constraints_on_every_knot=True, decay_factor=0.2):
        super().__init__(CPs, t_max, t_min, V_max, A_max, J_max, p, constraints_on_every_knot, decay_factor)

    def get_G_h(self):
        _G_1, h_1 = self._get_G_h()
        V_T, A_T = self.BSpline.get_A_transform(self.get_t(), self.p)
        V_CPs = (V_T @ (self.CPs.reshape(-1, 1))).reshape(-1, 3)
        A_CPs = (A_T @ (self.CPs.reshape(-1, 1))).reshape(-1, 3)

        n = self.t.shape[0]
        _G_2 = np.zeros((n+2, n))
        _G_3 = np.zeros((n+1, n))
        _G_4 = np.zeros((n, n))
        for i in range(n):
            _G_2[i:i+3, i] = -np.ones(3)
            _G_3[i:i+2, i] = -np.ones(2)
            _G_4[i:i+1, i] = -np.ones(1)

        G = np.vstack((_G_1, _G_2, _G_3, _G_4))

        h_2 = np.zeros((3, n+2))
        for i in range(self.n-1):
            _h_2 = 3 * np.abs(self.CPs[i+1]-self.CPs[i]) / self.V_max 
            h_2[:, i] = _h_2
        h_2 = -np.max(h_2, axis=0).reshape(-1, 1)
        
        h_3 = np.zeros((3, n+1))
        for i in range(self.n-2):
            _h_3 = 2 * np.abs(V_CPs[i+1]-V_CPs[i]) / self.A_max 
            h_3[:, i] = _h_3
        h_3 = -np.max(h_3, axis=0).reshape(-1, 1)
        
        h_4 = np.zeros((3, n))
        for i in range(self.n-3):
            _h_4 = np.abs(A_CPs[i+1]-A_CPs[i]) / self.J_max
            h_4[:, i] = _h_4
        h_4 = -np.max(h_4, axis=0).reshape(-1, 1)
        
        h = np.vstack((h_1, h_2, h_3, h_4))

        return G, h


class Knots_Optimizer_Iter_Linearization:
    def __init__(self, CPs, t_max, t_min, V_max, A_max, J_max, p, constraints_on_every_knot=False, alpha=0.5, decay_factor=0.2):
        self.CPs = CPs
        self.t_max = t_max
        self.t_min = t_min
        self.V_max = V_max
        self.A_max = A_max
        self.J_max = J_max
        self.n = CPs.shape[0]
        self.p = p
        self.alpha = alpha
        self.t = np.array([t_max/(self.n-p)]*(self.n-p))

        
        self.optimizer = Knots_Optimizer_base_Simple(CPs, t_max, t_min, V_max, A_max, J_max, p, constraints_on_every_knot, decay_factor)
        self.BSpline = self.optimizer.BSpline

        self.t_History = [self.t]


    def update_decay_factor(self, epoch):
        self.optimizer.update_decay_factor(epoch)
    
    def update_t(self, t):
        self.t = t
        self.optimizer.update_t(t)

    def update_CPs(self, CPs):
        self.CPs = CPs
        self.optimizer.update_CPs(CPs)

    def update_tmax(self, t):
        self.t_max = t[-1]
        self.optimizer.update_tmax(t)

    def update_tmin(self, t_min):
        self.t_min = t_min

    def get_t(self):
        t = self.t
        t = [sum(self.t[:i]) for i in range(self.t.shape[0]+1)]
        t = t[0:1] * self.p + t + t[-1:] * self.p

        return t

    def Iter(self):
        flag = True
        try:
            t = self.optimizer.optimize_knots()
            self.t_History.append(t)

            self.update_tmax(t)
            self.update_t(t)

        except:
            t = self.t_History[-1]
            flag = False

        return t, flag
    


if __name__ == "__main__":
    CPs = np.array([np.array([i, i, i]) for i in range(10)])
    t_max = 30
    t_min = 3.4
    V_max = 5
    A_max = 5
    J_max = 5
    p = 3

    optimizer = Knots_Optimizer_Iter_Linearization(CPs, t_max, t_min, V_max, A_max, J_max, p)
    G, h = optimizer.get_G_h()
    print(f"G shape: {G.shape}")
    print(f"h shape: {h.shape}")