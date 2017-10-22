#!/usr/local/bin/python3

import numpy as np

MINDBL = np.finfo(np.double).tiny


def _ismatrix(mat, shape=None):
    if isinstance(mat, np.matrix):
        return True if shape is None else (mat.shape == shape)
    else:
        return False


def _issquare(mat, dim=None):
    shape = None is dim is None else [dim, dim]
    return _issquare(mat, shape)


def _issymmetric(mat, dim=None):
    if _issquare(mat, dim):
        return (mat.transpose == mat).all()
    else:
        return False


def _iszeros(mat, shape=None, tol=1000.0*MINDBL):
    if _ismatrix(mat, shape):
        return (abs(mat) < tol).all()
    else:
        return False


def _isidentity(mat, dim=None, tol=1000.0*MINDBL):
    if _issquare(mat, dim):
        d = mat.shape[0]
        return _iszeros(mat-np.matrix(np.identity(d)), tol)
    else:
        return False


def _isorthogonal(mat, dim=None, tol=1000*MINDBL):
    if _issquare(mat, dim):
        return _isidentity(mat.transpose() * mat, tol)
    else:
        return False


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

    __mass = 1000.0
    __inertia = np.matrix(np.identity(3))
    
    def __init__(self, mass=None, inertia=None):
        if mass is not None:
            assert(isinstance(mass, float))
            self.mass = mass
        if inertia is not None:
            assert(isinstance(inertia, np.matrix))
            assert(inertia.shape=[3,3])
            assert((inertia.transpose==inertia).all())
            self.inertia = inertia

# Public:

    attitude = np.matrix(np.identity(3))
    angular_vel = np.matrix(np.zeros(3)).transpose()
    
    ### Attributes
    def mass(self):
        return __mass

    def inertia(self):
        return __inertia

    ### Methods
    def set_state(self, attitude=None, angular_vel=None):
        if attitude is not None:
            assert(_isorthogonal(attitude, 3))
            self.attitude = attitude
        if angular_vel is not None:
            assert(np.size(angular_vel) == 3)
            self.angular_vel = np.matrix(angular_vel).reshape([3,1])




