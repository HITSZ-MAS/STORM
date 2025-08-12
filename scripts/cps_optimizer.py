import math
import numpy as np
import scipy.linalg as linalg
from cvxopt import matrix, solvers

from BSpline import BSpline

class CPs_Optimizer:
    def __init__(self, t, p, init_guess):
        self.t = t
        self.p = p
        self.n = init_guess.shape[0]
        self.window_size = 2*p - 1
        self.BSpline = BSpline(t, p)
        self.V_Bspline = BSpline(t[1:-1], p-1)
        self.A_Bspline = BSpline(t[2:-2], p-2)
        self.J_Bspline = BSpline(t[3:-3], p-3)
        self.start_time = self.BSpline.t_min
        self.end_time = self.BSpline.t_max
        self.CPs = init_guess.reshape(-1, 3)
        self.V_T, self.A_T = self.BSpline.get_A_transform(t, p)

    def update(self, t, cps):
        self.t = t
        self.BSpline = BSpline(t, self.p)
        self.V_Bspline = BSpline(t[1:-1], self.p-1)
        self.A_Bspline = BSpline(t[2:-2], self.p-2)
        self.J_Bspline = BSpline(t[3:-3], self.p-3)
        self.start_time = t[0]
        self.end_time = t[-1]
        self.CPs = cps
        self.V_T, self.A_T = self.BSpline.get_A_transform(t, self.p)

    def update_t(self, t):
        self.t = t
        self.BSpline = BSpline(t, self.p)
        self.V_Bspline = BSpline(t[1:-1], self.p-1)
        self.A_Bspline = BSpline(t[2:-2], self.p-2)
        self.J_Bspline = BSpline(t[3:-3], self.p-3)
        self.start_time = t[0]
        self.end_time = t[-1]

    def Compute_Length(self):
        Trajectory = self.get_sampled_points(1000)
        Length = np.sum(np.linalg.norm(Trajectory[1:] - Trajectory[:-1], axis=1))
        return Length

    def get_Minmum_Jerk_Matrix_Sampled(self, nums):
        J_T = self.compute_J_T()
        M = self.compute_Jerk_Sampled_Matrix(nums)
        print(M.shape, J_T.shape)
        return J_T.T @ M.T @ M @ J_T

    def get_Minmum_Jerk_Matrix_Analytical(self):
        M = np.zeros((3*(self.n-3), 3*(self.n-3)))
        field = self.t[3:-3]
        for i in range(self.n-3):
            M[3*i:3*i+3, 3*i:3*i+3] = np.eye(3) * (field[i+1] - field[i])

        J_T = self.compute_J_T()

        return J_T.T @ M @ J_T
    
    def get_q(self):
        q = np.zeros((self.CPs.shape[0]*3, 1))
        return q

    def get_V_optimal_estimation(self):
        n_v = (self.CPs[-1] - self.CPs[0]) / np.linalg.norm(self.CPs[-1] - self.CPs[0]).reshape(-1,1)
        _v = n_v * 1.0 #最大可能速度
        _v = 1 / np.max(_v)
        return _v

    def compute_Jerk_Sampled_Matrix(self, nums):
        M = np.zeros((3*nums, 3*(self.n-3)))
        time_stamps = np.linspace(self.start_time, self.end_time, nums+2)
        for i in range(nums):
            weights, indicies = self.J_Bspline.get_Matrix4Point(time_stamps[i])
            for j, index in enumerate(indicies):
                M[3*i:3*i+3, 3*index:3*index+3] = weights[j] * np.eye(3)
            
        return M

    def compute_J_T(self):
        _, V_J_T = self.V_Bspline.get_A_transform(self.t[1:-1], self.p-1)
        return V_J_T @ self.V_T

    def get_Sampled_Points_Matrix(self, Weights):
        '''
        Parameters:
        -----------
        Weights: tuple of list(of np.array 1*(p+1))
        '''
        num1 = len(Weights[0])
        num2 = len(Weights[1])
        sample_num = num1 + num2 
        M = np.zeros((3*sample_num, 3*self.window_size))
        
        for i, weight in enumerate(Weights[0]):
            for j, w in enumerate(weight):
                M[3*i:3*i+3, 3*j:3*j+3] = w * np.eye(3)
        
        for i, weight in enumerate(Weights[1]):
            for j, w in enumerate(weight):
                M[3*(num1+i):3*(num1+i)+3, 3*(j+1):3*(j+1)+3] = w * np.eye(3)
        
        return M, num1, num2 # 3*sample_num x 3*self.window_size
    
    def get_Sampled_Points_Weights(self, Corridors, delta_t=0.01):
        '''
        Parameters:
        -----------
        Corridors: list of corridors
            Each corridor contains 1-2 constraints, each constraint is a pair of [P_i*3, P_i] matrices
            Where P_i*3 is the constraint matrix, P_i is the constraint vector
            All constraint vectors P_i must have the same dimension
        '''
        corridor_nums = [len(corridor) for corridor in Corridors]
        _intersection_indicies = [i for i, n in enumerate(corridor_nums) if n == 2 and i > 1 and i < len(corridor_nums)-2]
        intersection_indicies = []
        for k, i in enumerate(_intersection_indicies):
            if k == 0:
                intersection_indicies.append(i)
            else:
                if i - intersection_indicies[-1] >= self.window_size:
                    intersection_indicies.append(i)

        domain = self.BSpline.field
        Weights = []
        for i in intersection_indicies:
            intervals = domain[i-2:i+1]
            nums1 = math.ceil((intervals[1] - intervals[0]) / delta_t)
            nums2 = math.ceil((intervals[2] - intervals[1]) / delta_t)
            time_stamps1 = np.linspace(intervals[0], intervals[1], nums1+2)[1:-1]
            time_stamps2 = np.linspace(intervals[1], intervals[2], nums2+2)[1:-1]

            weights1 = [self.BSpline.get_Matrix4Point(t)[0] for t in time_stamps1]
            weights2 = [self.BSpline.get_Matrix4Point(t)[0] for t in time_stamps2]
            Weights.append((weights1, weights2))

        return Weights, intersection_indicies

    def _get_G_h(self, Corridors):
        G = []
        h = []
        Sub_Matrix_G = []
        Sub_Matrix_h = [] 
        for corridor in Corridors:   
            if len(corridor) == 2:
                Sub_Matrix_G.append(np.concatenate([corridor[0][0], corridor[1][0]], axis=0)) # p*3
                Sub_Matrix_h.append(np.concatenate([corridor[0][1], corridor[1][1]], axis=0)) # p
            else:
                Sub_Matrix_G.append(corridor[0][0])
                Sub_Matrix_h.append(corridor[0][1])

        _G = linalg.block_diag(*Sub_Matrix_G) 
        _h = -np.concatenate(Sub_Matrix_h, axis=0) 

        return _G, _h
   
    def get_G_h(self, Corridors):
        '''
        Parameters:
        -----------
        Corridors: list of corridors
            Each corridor contains 1-2 constraints, each constraint is a pair of [P_i*3, P_i] matrices
            Where P_i*3 is the constraint matrix, P_i is the constraint vector
            All constraint vectors P_i must have the same dimension
        '''
        if not Corridors:
            raise ValueError("Corridors cannot be empty")
        
        for i, corridor in enumerate(Corridors):
            if not (len(corridor) == 1 or len(corridor) == 2):
                raise ValueError(f"Corridor {i} must contain 1 or 2 constraints, currently contains {len(corridor)}")

        Weights, intersection_indicies = self.get_Sampled_Points_Weights(Corridors)
        
        G = []
        h = []
        Sub_Matrix_G = []
        Sub_Matrix_h = [] 
        for corridor in Corridors:   
            if len(corridor) == 2:
                Sub_Matrix_G.append(np.concatenate([corridor[0][0], corridor[1][0]], axis=0)) # p*3
                Sub_Matrix_h.append(np.concatenate([corridor[0][1], corridor[1][1]], axis=0)) # p
            else:
                Sub_Matrix_G.append(corridor[0][0])
                Sub_Matrix_h.append(corridor[0][1])

        for weights, i in zip(Weights, intersection_indicies):
            M, num1, num2 = self.get_Sampled_Points_Matrix(weights)
            c1 = Corridors[i][0][0]
            c2 = Corridors[i][1][0]
            M = linalg.block_diag(*([c1]*num1 + [c2]*num2)) @ M
            G_i = linalg.block_diag(*Sub_Matrix_G[i-2:i+3])
            G_i = np.concatenate([G_i, M], axis=0)
            Sub_Matrix_G[i] = G_i
            
            h_i = np.concatenate(Sub_Matrix_h[i-2:i+3], axis=0)
            h1 = Corridors[i][0][1] # p
            h2 = Corridors[i][1][1] # p
            h1 = np.concatenate([h1]*num1, axis=0)
            h2 = np.concatenate([h2]*num2, axis=0)
            h_i = np.concatenate([h_i, h1, h2], axis=0)
            Sub_Matrix_h[i] = h_i
        
        #########################################################
        del_indicies = []
        for i in intersection_indicies:
            del_indicies.extend([i-2, i-1, i+1, i+2])
        del_indicies = list(set(del_indicies))
        remain_indicies = [i for i in range(len(Sub_Matrix_G)) if i not in del_indicies]
        Sub_Matrix_G = [Sub_Matrix_G[i] for i in remain_indicies]
        Sub_Matrix_h = [Sub_Matrix_h[i] for i in remain_indicies]
        #########################################################

        G = linalg.block_diag(*Sub_Matrix_G)
        h = -np.concatenate(Sub_Matrix_h, axis=0)
        
        return G, h

    def get_A_b(self, Start_State, End_State, Start_Velocity, End_Velocity, n):
        '''
        Parameters:
        -----------
        Start_State: [3]
        End_State: [3]
        '''
        A = np.zeros((12, 3*n))
        b = np.zeros((12, 1))

        b[0:3, 0] = Start_State
        b[3:6, 0] = Start_Velocity
        b[6:9, 0] = End_State
        b[9:12, 0] = End_Velocity

        start_state_weights = self.BSpline.get_Matrix4Point(self.start_time)[0]
        end_state_weights = self.BSpline.get_Matrix4Point(self.end_time)[0]
        start_velocity_weights = self.V_Bspline.get_Matrix4Point(self.start_time)[0]
        end_velocity_weights = self.V_Bspline.get_Matrix4Point(self.end_time)[0]
        start_V_T = self.V_T[0:3*self.p, 0:3*(self.p+1)]
        end_V_T = self.V_T[-3*self.p:, -3*(self.p+1):]

        p_num = len(start_state_weights)
        for i, (s, e) in enumerate(zip(start_state_weights, end_state_weights)):
            A[0:3, 3*i:3*(i+1)] = s * np.eye(3)
            A[6:9, -3*(p_num-i):-3*(p_num-i-1) if p_num-i-1 > 0 else None] = e * np.eye(3)

        v_num = len(start_velocity_weights)
        T1 = np.zeros((3, 3*v_num))
        T2 = np.zeros((3, 3*v_num))
        for i, (s, e) in enumerate(zip(start_velocity_weights, end_velocity_weights)):
            T1[0:3, 3*i:3*(i+1)] = s * np.eye(3)
            T2[0:3, 3*i:3*(i+1)] = e * np.eye(3)
            # T2[0:3, -3*(v_num-i):-3*(v_num-i-1) if v_num-i-1 > 0 else None] = e * np.eye(3)
        T1 = T1 @ start_V_T
        T2 = T2 @ end_V_T
        A[3:6, 0:3*(self.p+1)] = T1
        A[9:12, -3*(self.p+1):] = T2
        
        return A, b

    def optimize_trajectory(self, 
                            Corridors, 
                            Start_State, 
                            End_State, 
                            Start_Velocity,
                            End_Velocity,
                            guidance_gradient=None
                            ):
        """
        Solve trajectory optimization problem
        
        Parameters: 
        -----------
        Corridors: list
            List of corridor constraints
        n: int
            Dimension of optimization variables
            
        Returns:
        --------
        dict
            Optimization result
        """
        n = self.n

        P = self.get_Minmum_Jerk_Matrix_Analytical() / 2
        P = matrix(P, (P.shape[0], P.shape[1]), 'd')
        
        if guidance_gradient is None:
            q = self.get_q().reshape(-1, 1)
            q = matrix(q, (q.shape[0], q.shape[1]), 'd')
        else:
            q = guidance_gradient
            q = matrix(q, (q.shape[0], q.shape[1]), 'd')
        
        G, h = self._get_G_h(Corridors)
        G = matrix(G, (G.shape[0], G.shape[1]), 'd')
        h = matrix(h, (h.shape[0], 1), 'd')
        
        A, b = self.get_A_b(Start_State, End_State, Start_Velocity, End_Velocity, n)
        A = matrix(A, (A.shape[0], A.shape[1]), 'd')
        b = matrix(b, (b.shape[0], 1), 'd')

        cps = self.CPs.reshape(-1, 1)
        init_guess = {'x': matrix(cps, (3*n,1), 'd')}

        try:
            solvers.options['show_progress'] = False
            sol = solvers.qp(P, q, G, h, A, b, initvals=init_guess)
            if sol['status'] != 'optimal':
                raise RuntimeError(f"Optimization failed, status: {sol['status']}")
            optimal_trajectory = np.array(sol['x']).reshape(n, 3)
            return optimal_trajectory
            
        except Exception as e:
            pass

        return None
        
    def get_sampled_points(self, nums=1000):
        nums = np.ceil((self.end_time - self.start_time) / 0.001).astype(int)
        time_stamps = np.linspace(self.start_time, self.end_time, nums+2)
        points = []
        for i in range(nums):
            weights, indicies = self.BSpline.get_Matrix4Point(time_stamps[i])
            CPs = self.CPs[indicies]
            points.append(weights @ CPs)
        points = np.array(points)

        return points
    
    def get_sampled_points_derivative(self, nums=1000):
        J_T = self.compute_J_T()
        nums = np.ceil((self.end_time - self.start_time) / 0.001).astype(int)
        time_stamps = np.linspace(self.start_time, self.end_time, nums+2)
        v_points = []
        a_points = []
        j_points = []
        v_cps = (self.V_T @ (self.CPs.reshape(-1, 1))).reshape(-1, 3)
        a_cps = (self.A_T @ (self.CPs.reshape(-1, 1))).reshape(-1, 3)
        j_cps = (J_T @ (self.CPs.reshape(-1, 1))).reshape(-1, 3)
        
        for i in range(nums):
            weights, indicies = self.V_Bspline.get_Matrix4Point(time_stamps[i])
            CPs = v_cps[indicies]
            v_points.append(weights @ CPs)
        v_points = np.array(v_points)

        for i in range(nums):
            weights, indicies = self.A_Bspline.get_Matrix4Point(time_stamps[i])
            CPs = a_cps[indicies]
            a_points.append(weights @ CPs)
        a_points = np.array(a_points)
        
        for i in range(nums):
            weights, indicies = self.J_Bspline.get_Matrix4Point(time_stamps[i])
            CPs = j_cps[indicies]
            j_points.append(weights @ CPs)
        j_points = np.array(j_points)

        return v_points, a_points, j_points

    def collision_check(self, trajectory, corridors):
        collision = []
        for i in range(trajectory.shape[0]):
            q = trajectory[i]
            if self.check_single_point(q, corridors):
                collision.append(True)
            else:
                collision.append(False)

        return collision
    
    def check_single_point(self, point, corridors):

        for corridor in corridors:
            if not self.check_single_point_single_corridor(point, corridor):
                return False
        return True
    
    def check_single_point_single_corridor(self, point, corridor):

        n_vecs = corridor[0]
        p_vecs = corridor[1]
        for j in range(n_vecs.shape[0]):
            n = n_vecs[j]
            p = p_vecs[j]
            if np.dot(n, point) + p > 0:
                return True
        return False 


if __name__ == "__main__":
    T = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    k = 3
    optimizer = CPs_Optimizer(T, k, np.zeros((6, 3)))
    print("CPs_Optimizer initialized successfully")