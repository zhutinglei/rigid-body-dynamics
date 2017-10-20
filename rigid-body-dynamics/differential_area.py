#!/usr/local/bin/python3

import numpy as np

class DifferentialArea:
    """
    Require position, normal, angular_velocity be 3-d vector, and normal should 
    be unit vector

    Parameters:
    ==========
    position, angular_velocity: 3-d vector (double)
    normal, tangent, binormal: 3-d unit vector (double), local frame

    """
    def __init__(self, position=None, angular_velocity=None, normal=None):
	    if position is None:
            self.position = np.array([-1.0, 0.0, 0.0])
        else: 
            self.position = position
        if angular_velocity is None:
            self.angular_velocity = np.array([0.0, 0.0, 1.0])
        else:
            self.angular_velocity = angular_velocity
        if normal is None:
            self.normal = self.position
        else:
            self.normal = normal/np.linalg.norm(normal)
        
        self.tangent = np.cross(angular_velocity, position)
        assert(self.tangent != np.zeros([3])
        self.tangent /= np.linalg.norm(self.tangent)
        self.binormal = np.cross(self.normal, self.tangent)


def test():
    diff_area = DifferentialArea()


if __name__ == '__main__':
    
    test()
