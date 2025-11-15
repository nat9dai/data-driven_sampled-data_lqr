import numpy as np
from scipy import linalg

class System():
    def __init__(self, A, B, n=None, m=None):
        self.A = A
        self.B = B

        self.n = n if n is not None else A.shape[0]
        self.m = m if m is not None else B.shape[1]

    def compute_true_Ad_Bd(self, h):
        """Computes exact discrete-time matrices Ad and Bd."""
        # Lemma 10.5.1 of Optimal Sampled-data Control Systems by Chen and Francis
        M = np.block([[self.A, self.B], [np.zeros((self.m, self.n + self.m))]])
        E = linalg.expm(M*h)
        return E[:self.n, :self.n], E[:self.n, self.n:]
    
    def step(self, x_k, u_k, h):
        """Performs one step of discrete-time dynamics."""
        Ad, Bd = self.compute_true_Ad_Bd(h)
        return Ad @ x_k + Bd @ u_k

class CartPole(System):
    def __init__(self, parameters: dict):
        g = parameters.get('g', 9.81)
        l = parameters.get('l', 1.0)
        m_p = parameters.get('m_p', 0.2)
        m_c = parameters.get('m_c', 0.5)

        A = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, m_p*g/m_c, 0.0, 0.0],
            [0.0, (m_p + m_c)*g/(m_c*l), 0.0, 0.0]
        ])

        B = np.array([[0.0], [0.0], [1.0/m_c], [1.0/(m_c*l)]])
        
        super().__init__(A, B)