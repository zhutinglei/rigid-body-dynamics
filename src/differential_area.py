#!/usr/local/bin/python3

import numpy as np

class DifferentialArea:
    position = np.array([-1.0, 0.0, 0.0], dtype=float)
    normal = position
    angular_velocity = np.array([0.0, 0.0, 1.0], dtype=float)
    tangent = np.zeros([3])
    binormal = np.zeros([3])

    def __init__(self, position=None, normal=None, angular_velocity=None):
        self.position = np.array([-1.0, 0.0, 0.0], dtype=float) if position == None else position
        


def test():
    diff_area = DifferentialArea()


if __name__ == '__main__':
    
    test()
