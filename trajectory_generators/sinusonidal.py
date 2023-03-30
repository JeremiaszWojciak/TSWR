import numpy as np
from trajectory_generators.trajectory_generator import TrajectoryGenerator


class Sinusoidal(TrajectoryGenerator):
    def __init__(self, A, omega, fi):
        self.A = A
        self.omega = omega
        self.fi = fi

    def generate(self, t):
        q = self.A * np.sin(self.omega * t + self.fi)
        q_dot = self.A * self.omega * np.cos(self.omega * t)
        q_ddot = - self.A * self.omega**2 * np.sin(self.omega * t)
        # print('A: ', self.A)
        # print('omega: ', self.omega)
        # print('fi: ', self.fi)
        # print('q: ', q)
        # print('q_dot: ', q_dot)
        # print('q_ddot: ', q_ddot)
        return q, q_dot, q_ddot
