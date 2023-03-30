import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kd = -1.0
        self.Kp = -1.0

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x
        # v = q_r_ddot
        v = q_r_ddot + self.Kd * (np.array([q1_dot, q2_dot]) - q_r_dot) + self.Kp * (np.array([q1, q2]) - q_r)
        u = self.model.M(x) @ v[:, np.newaxis] + self.model.C(x) @ np.array([q1_dot, q2_dot])[:, np.newaxis]
        # print('M: ', self.model.M(x))
        # print('q_r_ddot: ', q_r_ddot)
        # print('M @ q_r_ddot: ', self.model.M(x) @ q_r_ddot)
        # print('C: ', self.model.C(x))
        # print('[q1_dot, q2_dot]: ', np.array([q1_dot, q2_dot]))
        # print('C @ [q1_dot, q2_dot]: ', self.model.C(x) @ np.array([q1_dot, q2_dot]))
        return u
