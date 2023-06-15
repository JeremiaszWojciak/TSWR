import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from scipy.integrate import odeint

from controllers.adrc_controller import ADRController

from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.001
end = 5

# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))
# traj_gen = Poly3(np.array([0., 0.]), np.array([pi/4, pi/6]), end)

b_est_1 = 3
b_est_2 = 3
kp_est_1 = 60
kp_est_2 = 60
kd_est_1 = 10
kd_est_2 = 10
p1 = 200
p2 = 200

q0, qdot0, _ = traj_gen.generate(0.)
q1_0 = np.array([q0[0], qdot0[0]])
q2_0 = np.array([q0[1], qdot0[1]])
controller = ADRController(Tp, params=[[b_est_1, kp_est_1, kd_est_1, p1, q1_0],
                                       [b_est_2, kp_est_2, kd_est_2, p2, q2_0]])

Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

eso1 = np.array(controller.joint_controllers[0].eso.states)
eso2 = np.array(controller.joint_controllers[1].eso.states)

plt.subplot(221)
plt.plot(T, eso1[:, 0], label="q1_hat")
plt.plot(T, Q[:, 0], 'r', label="q1")
plt.legend()
plt.subplot(222)
plt.plot(T, eso1[:, 1], label="q1_dot_hat")
plt.plot(T, Q[:, 2], 'r', label="q1_dot")
plt.legend()
plt.subplot(223)
plt.plot(T, eso2[:, 0], label="q2_hat")
plt.plot(T, Q[:, 1], 'r', label="q2")
plt.legend()
plt.subplot(224)
plt.plot(T, eso2[:, 1], label="q2_dot_hat")
plt.plot(T, Q[:, 3], 'r', label="q2_dot")
plt.legend()
plt.show()

plt.subplot(221)
plt.plot(T, Q[:, 0], 'r', label="q1")
plt.plot(T, Q_d[:, 0], 'b', label="q1_d")
plt.legend()
plt.subplot(222)
plt.plot(T, Q[:, 1], 'r', label="q2")
plt.plot(T, Q_d[:, 1], 'b', label="q2_d")
plt.legend()
plt.subplot(223)
plt.plot(T, u[:, 0], 'r', label="u1")
plt.plot(T, u[:, 1], 'b', label="u2")
plt.legend()
plt.show()