import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],
                      [self.b],
                      [0]])
        L = np.array([[3 * p],
                      [3 * p ** 2],
                      [p ** 3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)
        self.model = ManipulatorModel(Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        B = np.array([[0],
                     [self.b],
                     [0]])
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, i):
        ### TODO implement ADRC
        z_hat = self.eso.get_state()
        f_hat = z_hat[2]
        v = q_d_ddot + self.kd * (q_d_dot - z_hat[1]) + self.kp * (q_d - x[0])
        u = (v - f_hat) / self.b

        if i == 1:
            M_inv = np.linalg.inv(self.model.M([0.0, x[0], 0.0, x[1]]))
            new_b = M_inv[i, i]
            self.set_b(new_b)
            print(new_b)

        self.eso.update(x[0], u)
        return u
