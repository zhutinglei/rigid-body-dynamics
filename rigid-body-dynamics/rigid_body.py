#!/usr/local/bin/python3

import numpy as np

MINDBL = np.finfo(np.double).tiny


def _ismatrix(mat, shape=None):
    if not isinstance(mat, np.matrix):
        return False
    else if shape is not None:
        return mat.shape = shape
    else:
        return true


def _issymmetric(mat, dim=None):
    shape = None if dim is None else [dim, dim]
    if _ismatrix(mat, shape):
        return (mat.transpose == mat).all()
    else:
        return False


def _isidentity(mat, dim=None, tol=1000.0*MINDBL):
    shape = None if dim is None else [dim, dim]
    if _ismatrix(mat, shape):
        identity = np.matrix(np.identity(mat.shape[0]))
        return (abs(mat-identity)<tol).all()
    else:
        return False

def _isorthogonal(mat, dim=None, tol=1000*MINDBL):
    shape = None if dim is None else [dim, dim]
    if _ismatrix(mat, shape):
        identity = mat.transpose * mat


        
    
        

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
    mass = 1000.0
    inertia = np.matrix(np.identity(3))
    attitude = np.matrix(np.identity(3))
    angular_vel = np.matrix(np.zeros(3)).transpose()
    
    def __init__(self, mass=None, inertia=None):
        if mass is not None:
            assert(isinstance(mass, float))
            self.mass = mass
        if inertia is not None:
            assert(isinstance(inertia, np.matrix))
            assert(inertia.shape=[3,3])
            assert((inertia.transpose==inertia).all())
            self.inertia = inertia
    # Attributes

    # Methods
    def init_state(self, attitude=None, angular_vel=None):
        if attitude is not None:



