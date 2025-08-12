import numpy as np
import json
import argparse

from cps_optimizer import CPs_Optimizer
from knots_optimizer import Knots_Optimizer_Iter_Linearization

def parse_args():
    parser = argparse.ArgumentParser(description='Iterative Optimizer')
    parser.add_argument('--alpha', type=float, default=0, help='alpha')
    parser.add_argument('--decay_factor', type=float, default=0, help='decay_factor')
    parser.add_argument('--beta', type=float, default=0, help='beta')
    parser.add_argument('--regular_factor', type=float, default=0.1, help='regular_factor')
    parser.add_argument('--early_stop', type=bool, default=True, help='early_stop')
    parser.add_argument('--A_max', type=float, default=7, help='A_max')
    parser.add_argument('--J_max', type=float, default=20, help='J_max')
    return parser.parse_args()

class Iter_Optimizer:
    def __init__(self, 
                 CPs, 
                 p,
                 t_max, 
                 V_max, 
                 A_max,
                 J_max,
                 Corridors,
                 Start_State, 
                 End_State, 
                 Start_Velocity,
                 End_Velocity,
                 constraints_on_every_knot=False,
                 regular_factor=0.1,
                 alpha=0.5,
                 decay_factor=0.2,
                 beta=0.1
                 ):
        '''
        CPs: np.array of shape (n, 3)
        t_max: maximum time
        V_max: maximum velocity, np.array of shape (3,)
        A_max: maximum acceleration, np.array of shape (3,)
        p: degree of the B-spline
        '''
        self.CPs = CPs.reshape(-1, 3)
        self.V_max = V_max
        self.A_max = A_max
        self.J_max = J_max
        self.t_max = t_max
        self.t_min = self.compute_t_min(Corridors)
        self.Corridors = Corridors
        self.Start_State = Start_State
        self.End_State = End_State
        self.Start_Velocity = Start_Velocity
        self.End_Velocity = End_Velocity
        self.p = p

        self.Knots_Optimizer = Knots_Optimizer_Iter_Linearization(self.CPs, t_max, self.t_min, V_max, A_max, J_max, p, constraints_on_every_knot, alpha, decay_factor)
        self.t = self.Knots_Optimizer.get_t()
        self.CPs_Optimizer  = CPs_Optimizer(self.t, p, self.CPs)

        self.CPs_History = [self.CPs]
        self.knots_History = [self.t]
        self.cps_delta_History = []

    def compute_t_min(self, Corridors):
        intersection_indicies = [i if len(Corridors[i]) == 2 else None for i in range(len(Corridors))]
        intersection_indicies = [i for i in intersection_indicies if i is not None]
        t_min = np.sum([np.linalg.norm(self.CPs[intersection_indicies[i]] - self.CPs[intersection_indicies[i+1]])/self.V_max for i in range(len(intersection_indicies)-1)])

        return t_min

    def update_tmin(self):
        t_min = self.compute_t_min(self.Corridors)
        self.t_min = t_min
        self.Knots_Optimizer.update_tmin(t_min)

    def Iter(self, terminate=False, CPs_terminate=False, knots_terminate=False):
        if CPs_terminate:
            self.Knots_Optimizer.Iter()
            t = self.Knots_Optimizer.get_t()
            self.CPs_Optimizer.update_t(t)
            self.Knots_Optimizer.update_tmax(t)

            self.knots_History.append(t)

        Optimized_CPs = self.CPs_Optimizer.optimize_trajectory(
            self.Corridors,
            self.Start_State,
            self.End_State,
            self.Start_Velocity,
            self.End_Velocity,
        )
        if Optimized_CPs is None:
            return None, False
        
        self.CPs = Optimized_CPs
        self.CPs_History.append(Optimized_CPs)

        if terminate:
            return Optimized_CPs
        
        self.Knots_Optimizer.update_CPs(Optimized_CPs)
        _, flag = self.Knots_Optimizer.Iter()
        t = self.Knots_Optimizer.get_t()
        self.CPs_Optimizer.update(t, Optimized_CPs)
        self.Knots_Optimizer.update_tmax(t)

        self.knots_History.append(t)

        return Optimized_CPs, flag

    def Optimize(self, max_iter=100, tol_CPs=5e-2, tol_knots=5e-2):
        cps_terminate = False
        for i in range(max_iter):
            print(f"==============Iter_Epoch_{i} Begin==============")
            Optimized_CPs, flag = self.Iter(cps_terminate) 
            if Optimized_CPs is None:
                return None, None, i+1
            self.Knots_Optimizer.update_decay_factor(i+1)
            CPs_delta = np.linalg.norm(self.CPs_History[-1] - self.CPs_History[-2], ord=np.inf)
            delta_knots = np.array(self.knots_History[-1]) - np.array(self.knots_History[-2])
            knots_delta = np.abs(delta_knots[-1])
            print(CPs_delta, knots_delta)
            print(f"==============Iter_Epoch_{i} Done==============")
            if i == max_iter-1:
                print("==============Max Iter Reached==============")
                raise Exception("Max Iter Reached")
            if knots_delta < tol_knots:
                break
        return self.CPs_History[-1], self.knots_History[-1], i+1

    def Optimize_with_Defeat_Stopping(self, max_iter=100, min_iter=5, tol_CPs=5e-2, tol_knots=5e-2):
        cps_terminate = False
        for i in range(max_iter):
            print(f"==============Iter_Epoch_{i} Begin==============")
            Optimized_CPs, flag = self.Iter(cps_terminate) 
            if Optimized_CPs is None: 
                return None, None, i+1
            if flag:
                self.Knots_Optimizer.update_decay_factor(i+1)
                CPs_delta = np.linalg.norm(self.CPs_History[-1] - self.CPs_History[-2], ord=np.inf)
                delta_knots = np.array(self.knots_History[-1]) - np.array(self.knots_History[-2])
                knots_delta = np.abs(delta_knots[-1])
                print(CPs_delta, knots_delta)
                print(f"==============Iter_Epoch_{i} Done==============")
            else:
                if i > min_iter-1:
                    break
                else:
                    raise Exception("Knots Optimizer Failed")

            if i == max_iter-1:
                print("==============Max Iter Reached==============")
                raise Exception("Max Iter Reached")
            
            if knots_delta < tol_knots:
                break

        return self.CPs_History[-1], self.knots_History[-1], i+1

    def Compute_Length(self):
        Trajectory = self.CPs_Optimizer.get_sampled_points(1000)
        Length = np.sum(np.linalg.norm(Trajectory[1:] - Trajectory[:-1], axis=1))
        return Length

    def compute_Jerk(self):
        jerk_matrix = self.CPs_Optimizer.get_Minmum_Jerk_Matrix_Analytical()
        # jerk_matrix = self.CPs_Optimizer.get_Minmum_Jerk_Matrix_Sampled(100)
        cps = self.CPs.reshape(-1, 1)

        return cps.T @ jerk_matrix @ cps

    def compute_collision(self, corridors):
        sampled_points = self.CPs_Optimizer.get_sampled_points(1000)
        collision = self.CPs_Optimizer.collision_check(sampled_points, corridors)
        collision_count = np.sum(collision)
        avg_collision = collision_count / 1000

        return avg_collision

    def compute_kinodynamic_violation(self):
        v_sampled_points, a_sampled_points = self.CPs_Optimizer.get_sampled_points_derivative(1000)
        v_violation = np.mean(np.max(np.abs(v_sampled_points), axis=1) > self.V_max)
        a_violation = np.mean(np.max(np.abs(a_sampled_points), axis=1) > self.A_max)

        return v_violation, a_violation
    