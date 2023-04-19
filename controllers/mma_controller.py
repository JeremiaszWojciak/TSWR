import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        model_1 = ManipulatorModel(Tp)
        model_1.m3 = 0.1
        model_1.r3 = 0.05
        # II:  m3=0.01, r3=0.01
        model_2 = ManipulatorModel(Tp)
        model_2.m3 = 0.01
        model_2.r3 = 0.01
        # III: m3=1.0,  r3=0.3
        model_3 = ManipulatorModel(Tp)
        model_3.m3 = 1.0
        model_3.r3 = 0.3
        self.models = [model_1, model_2, model_3]
        self.i = 0
        self.Tp = Tp
        self.u_prev = np.array([[0], [0]])
        self.x_prev = np.array([0, 0, 0, 0])
        self.x_m_dot_prev = np.zeros((3, 4))
        self.Kd = 1.0
        self.Kp = 10.0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        err = [0, 0, 0]
        for i, model in enumerate(self.models):
            x_m_dot = model.x_dot(self.x_prev, self.u_prev)
            # print('x_m_dot: ', x_m_dot)
            q_m_dot_1 = self.x_prev[2] + ((x_m_dot[2] + self.x_m_dot_prev[i][2]) / 2 * self.Tp)
            q_m_dot_2 = self.x_prev[3] + ((x_m_dot[3] + self.x_m_dot_prev[i][3]) / 2 * self.Tp)
            self.x_m_dot_prev[i] = x_m_dot.ravel()
            print(q_m_dot_1)
            err[i] = (abs(x[2] - q_m_dot_1) + abs(x[3] - q_m_dot_2)) / 2

        # print(err)
        self.i = np.argmin(err)
        # print(self.i)
        self.x_prev = x

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q1, q2, q1_dot, q2_dot = x
        # v = q_r_ddot + self.Kd * (q_r_dot - np.array([q1_dot, q2_dot])) + self.Kp * (q_r - np.array([q1, q2]))
        v = q_r_ddot
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ np.array([q1_dot, q2_dot])[:, np.newaxis]
        self.u_prev = u
        return u
