#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import checks as ck

class RigidBody:
    """
        Define the status of a rigid body, the center of mass is fixed 
    as origin.

    Parameters:
    ==========
    mass: float 
        unit = kg, default = 1000.0 
    inertia: `numpy.matrix`, shape=[3,3] 
        unit = kg*m^2, default = np.matrix(np.identity), inertia is a 
    symmetric matrix represented in the body-fixed frame (BFF), so it is
    constant for a specific rigid body.
    
    attitude: `numpy.matrix`, shape=[3,3] 
        attitude is an orthongonal matrix, i.e. inverse = transpose, the
    columns are coordinates of the axes of BFF in translation frame (TF)
    angular_vel: `numpy.matrix`, shape=[3,1] angular velocity of the 
    rigid body, given in BFF. 
    
    """

# Private: 

    __angle_types = ('zxz', 'xyz')
    __mass = 1000.0
    __inertia = np.matrix(np.identity(3))
    
    def __init__(self, mass=None, inertia=None):
        if mass is not None:
            assert(isinstance(mass, float))
            self.mass = mass
        if inertia is not None:
            assert(ck.issymmetric(inertia, 3))
# Public:

    attitude = np.matrix(np.identity(3))
    angular_vel = np.matrix(np.zeros(3)).transpose()
    
    ### Attributes
    def mass(self):
        return __mass

    def inertia(self):
        return __inertia

    def get_euler_angles(self, angle_type):
        assert(angle_type in __angle_types)
        ### TODO 
        pass

    ### Methods
    def set_state(self, attitude=None, angular_vel=None):
        if attitude is not None:
            assert(ck.isorthogonal(attitude, 3))
            self.attitude = attitude
        if angular_vel is not None:
            assert(np.size(angular_vel) == 3)
            self.angular_vel = np.matrix(angular_vel).reshape([3,1])
    
    def set_attitude_from_angles(self, angles, angle_type):
        assert(angle_type in __angle_types)
        # TODO
        pass

    def free_propagate(self, attitude=None, angular_vel=None, time_delta=None):
        """
            Propagate rigid body in Euler case, i.e. no external torque

        Parameters:
        ==========


        Return:
        =======
        """
        self.set_state(attitude, angular_vel)
        if time_delta is None:
            return self
        # TODO
        pass


