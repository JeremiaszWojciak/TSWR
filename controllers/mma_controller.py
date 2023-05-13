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
        self.u_prev = np.zeros((2, 1))
        self.x_prev = np.zeros(4)
        self.Kd = 20.0
        self.Kp = 50.0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        err = [0, 0, 0]
        for i, model in enumerate(self.models):
            x_m_dot = model.x_dot(self.x_prev, self.u_prev)
            x_m = self.x_prev + self.Tp * x_m_dot.flatten()
            err[i] = (abs(x[2] - x_m[2]) + abs(x[3] - x_m[3])) / 2

        self.i = np.argmin(err)
        self.x_prev = x

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q1, q2, q1_dot, q2_dot = x
        v = q_r_ddot + self.Kd * (q_r_dot - np.array([q1_dot, q2_dot])) + self.Kp * (q_r - np.array([q1, q2]))
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ np.array([q1_dot, q2_dot])[:, np.newaxis]
        self.u_prev = u
        return u
