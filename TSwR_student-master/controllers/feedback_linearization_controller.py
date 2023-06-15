import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp)
        self.Kd = 1.0
        self.Kp = 10.0

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q1, q2, q1_dot, q2_dot = x

        # v = q_r_ddot
        v = q_r_ddot + self.Kd * (q_r_dot - np.array([q1_dot, q2_dot])) + self.Kp * (q_r - np.array([q1, q2]))
        u = self.model.M(x) @ v[:, np.newaxis] + self.model.C(x) @ np.array([q1_dot, q2_dot])[:, np.newaxis]
        return u
