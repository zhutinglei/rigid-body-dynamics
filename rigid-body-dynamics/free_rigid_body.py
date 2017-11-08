#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import scipy.integrate as integrate


def ode_free_rigid_body(state, time, inertia):
    j = np.matrix(inertia)
    a = np.matrix(state[0:9].reshape([3,3]))
    w = np.matrix(state[9:12]).transpose()
    w_crs = np.cross(np.eye(3), w.getA1())
    dw = - np.linalg.inv(j) * w_crs * j * w
    da = a * w_crs * a
    dw = np.array(dw)
    da = np.array(da)
    return np.append(da, dw)


if __name__ == "__main__":
    print('Free rigid body dynamics')
    inertia = np.diag([1., 2., 3.])
    attitude = np.eye(3)
    angular_vel = np.array([1.5, 0.3, 2.0])
    state = np.append(attitude, angular_vel)

    print("state = ", state, type(state))
    t0 = 0.0
    t1 = 100.0
    n_steps = 100
    dt = (t1-t0)/n_steps
    times = np.arange(t0, t1, dt, dtype=float)
    print('start integrating')
    result = odeint(ode_free_rigid_body, state, times, args=(inertia,))
    print(result[:, 9:12])

