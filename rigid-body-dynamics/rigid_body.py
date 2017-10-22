#!/usr/local/bin/python3

import numpy as np

class RigidBody:
    """
    Define the status of a rigid body

    Parameters:
    ==========
    inertia: Ndarray, size=3

    """
    inertia = np.zeros([3])
    inertia_axes = np.identity(3)



