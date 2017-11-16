
# -*- encoding: utf-8 -*-

import numpy as np
from pyquaternion import Quaternion
from scipy.integrate import odeint
import scipy.integrate as integrate
import astropy.constants as const
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def gravity_gradient_torque(earth_center_position, inertia):
    """
    Return gravity gradient torque due to Earth

    Params:
    =======
    earth_center_position: `numpy.array`, shape=(3,)
        Defined in body-fixed frame
    inertia: `numpy.array`, shape=(3,3)
        A symmetric matrix, represent the body's moment of inertia, in
    body-fixed frame.

    Return:
    =======
    torque: `numpy.array`, shape=(3,)
        Gravity gradient torque, first term in expansion, in body-fixed
    frame.
    """
    gm = const.GM_earth.to_value()  # unit = m3/s2 
    r = earth_center_position
    torque = 3.0 * gm * np.norm(r)**(-5) * np.cross(r, np.matmul(inertia, r))
    return torque
    

def ode_rigid_body_1(state, time, inertia, earth_pos):
    """
    ODE for integration, use quaternion to represent attitude.

    Params:
    =======
    state: `numpy.array`, shape=(7,)
        state[0:4] represent the attitude of body-fixed frame relative
    to the translational frame through quaternion, i.e.
    q = Quaternion(state[0:4]), suppose v_bff = (x, y, z) is a vector
    represented in body-fixed frame, and its coordinates in 
    translational frame v_tf = q.rotate(v_bff), inversily, 
    v_bff = q.inverse.rotate(v_tf) or equivalently,
    v_bff = (q**-1).rotate(v_tf)
        state[4:7] represent the angular velocity in body-fixed frame 

    time: `float`

    inertia: `numpy.array`, shape=(3,3)
        A symmetric matrix, represented in body-fixed frame

    earth_pos: `numpy.array`, shape=(3,)
        A 3d vector, from satellite center to Earth center, represented
    in translational frame.
    """
    q = Quaternion(state[0:4])
    w = state[4:7]
    r = q.inverse.rotate(earth_pos)
    torque_free = - np.cross(w, np.matmul(inertia, w))
    # torque_grav = gravity_gradient_torque(r, inertia)
    torque_grav = np.zeros(3)
    dw = np.matmul(np.linalg.inv(inertia), torque_grav - torque_free)
    dq = 0.5 * q * Quaternion(scaler=0, vector=w)

    return np.append(dq.elements, dw)


if __name__ == "__main__":
    print('Rigid body dynamics')
    inertia = np.diag([1., 2., 3.])
    q = Quaternion(matrix=np.eye(3))
    w = np.array([1.0, 0.3, 2.0])
    state = np.append(q.elements, w)

    print("state = ", state, type(state))
    t0 = 0.0
    t1 = 100.0
    n_steps = 150
    dt = (t1-t0)/n_steps
    times = np.arange(t0, t1, dt, dtype=float)
    print('start integrating')
    result = odeint(ode_rigid_body_1, state, times, args=(inertia, np.zeros(3)))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(result[:,4], result[:, 5], result[:,6], 'k.')
    plt.show()
 
