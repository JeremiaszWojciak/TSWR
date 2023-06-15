import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
# from models.ideal_model import IdealModel
from models.manipulator_model import ManipulatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([[3 * p[0], 0],
                           [0, 3 * p[1]],
                           [3 * p[0] ** 2, 0],
                           [0, 3 * p[1] ** 2],
                           [p[0] ** 3, 0],
                           [0, p[1] ** 3]])
        W = np.zeros((2, 6))
        W[0:2, 0:2] = np.eye(2)
        self.A = np.zeros((6, 6))
        self.A[0:4, 2:6] = np.eye(4)
        self.B = np.zeros((6, 2))
        self.eso = ESO(self.A, self.B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot])
        self.A[2:4, 2:4] = - np.linalg.inv(self.model.M(x)) @ self.model.C(x)
        self.B[2:4, :] = np.linalg.inv(self.model.M(x))

        self.eso.A = self.A
        self.eso.B = self.B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q1, q2, q1_dot, q2_dot = x
        z_hat = self.eso.get_state()
        q_hat = z_hat[0:2]
        q_dot_hat = z_hat[2:4]
        f_hat = z_hat[4:]

        v = q_d_ddot + self.Kd @ (q_d_dot - q_dot_hat) + self.Kp @ (q_d - np.array([q1, q2]))
        u = self.model.M(z_hat[0:4]) @ (v - f_hat) + self.model.C(z_hat[0:4]) @ q_dot_hat

        self.update_params(q_hat, q_dot_hat)
        self.eso.update(np.array([[q1], [q2]]), u[:, np.newaxis])
        return u



