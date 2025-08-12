import numpy as np

class BSpline_Coeff_Matrix:
    def __init__(self, t, k):
        self.t = t
        self.k = k
        self.Coeffs = []
        self.get_Coeffs(k)
        # print("==========Coeffs Constructed==========")
        # print("len(self.Coeffs): ", len(self.Coeffs))

    def compute_d0_d1(self, i, j, k):
        """
        Compute d_{0,j} and d_{1,j} based on the given formula.
        """
        t = self.t
        d0_j = (t[i] - t[j]) / (t[j + k - 1] - t[j]) if t[j + k - 1] - t[j] != 0 else 0
        d1_j = (t[i + 1] - t[i]) / (t[j + k - 1] - t[j]) if t[j + k - 1] - t[j] != 0 else 0
        return d0_j, d1_j

    def construct_matrix_Mk(self, i, k):
        """
        Recursively construct the matrix M^k(i) based on the given formula.
        """
        if k == 1:
            return np.array([[1]])  # Base case: M^1(i) = [1]
        
        # Recursive case: k > 1
        M_k_minus_1 = self.construct_matrix_Mk(i, k - 1)  # Get M^{k-1}(i)
        n = M_k_minus_1.shape[0]  # Size of M^{k-1}(i)
        
        # Initialize the coefficient matrices A and B
        A = np.zeros((k - 1, k))
        B = np.zeros((k - 1, k))
        
        # Fill the coefficient matrices A and B
        for j in range(i - k + 2, i + 1):
            d0_j, d1_j = self.compute_d0_d1(i, j, k)
            idx = j - (i - k + 2)
            
            # Fill matrix A
            A[idx, idx] = 1 - d0_j
            A[idx, idx + 1] = d0_j 
            
            # Fill matrix B
            B[idx, idx] = - d1_j
            B[idx, idx + 1] = d1_j 
        
        # Construct M^k(i) by stacking and multiplying
        M_k = (
            np.vstack([M_k_minus_1, np.zeros((1, k-1))]) @ A +
            np.vstack([np.zeros((1, k-1)), M_k_minus_1]) @ B
        )
        
        return M_k
    
    def get_Coeffs(self, k):
        for i in range(k-1, len(self.t)-k):
            self.Coeffs.append(self.construct_matrix_Mk(i, k))


class BSpline:
    def __init__(self, t, p):
        self.t = t
        self.p = p
        self.n = len(t) - p - 1
        self.t_min = t[p]
        self.t_max = t[-p-1]
        self.field = t[p:-p] if p > 0 else t
        self.BM = BSpline_Coeff_Matrix(t, p+1)

    def get_V_transform(self, t, p):
        n = len(t) - p - 1
        V_T = np.zeros((3*(n-1), 3*n))

        for i in range(n-1):
            _s = p / (t[i+p+1] - t[i+1])
            V_T[3*i:3*i+3, 3*i:3*i+3] = -_s * np.eye(3)
            V_T[3*i:3*i+3, 3*i+3:3*i+6] = _s * np.eye(3)

        return V_T # 3*(n-1) x 3*n

    def get_A_transform(self, t, p):
        _V_T = self.get_V_transform(t, p)
        t = t[1:-1]
        p -= 1
        _A_T = self.get_V_transform(t, p)

        return _V_T, _A_T @ _V_T

    def get_Matrix4Point(self, t):
        assert t >= self.t_min and t <= self.t_max, "t is out of range"

        field = self.t[self.p:-self.p] if self.p > 0 else self.t
        field[-1] += 0.1 # to process the last point
        i = np.searchsorted(field, t, side='right') - 1 # t_i <= t < t_{i+1} [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]

        M_i = self.BM.Coeffs[i]
        interval = self.field[i:i+2] # [t_i, t_{i+1}]
        _u = (t - interval[0]) / (interval[1] - interval[0])
        u = np.array([_u ** j for j in range(self.p+1)])

        # return the result and the indices of the control points
        return u @ M_i, [i+j for j in range(self.p+1)]

if __name__ == "__main__":
    T = [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
    k = 4
    bspline = BSpline(T, k-1) 
    print(bspline.get_Matrix4Point(0))
    print(bspline.get_Matrix4Point(4))
    print(bspline.derivate_transform[1].shape)
