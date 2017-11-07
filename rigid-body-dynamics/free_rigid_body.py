#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-

import numpy as np
import scipy.integrate as integrate


def ode_free_rigid_body(time, state, inertia):
    j = inertia
    a = state[0:9].reshape([3,3])
    w = state[9:12]
    w_crs = np.cross(np.eye(3), w)
    dw = - np.linalge.inv(j) * w_crs * j * w
    da = a * w_crs * a
    return np.append(da, dw)


def propagate(state, inertia, t0, t1, n_steps):
    r = integrate.ode(ode_free_rigid_body).set_integrator('zvode', method='bdf')
    r.set_initial_value(state, t0).set_f_params(inertia)
    dt = (t1 - t0)/n_steps
    times = [t0]
    states = [state]
    while r.successful() and r.t < t1:
        t = min(r.t + dt, t1)
        times.append(t)
        states.append(r.integrate(t)
