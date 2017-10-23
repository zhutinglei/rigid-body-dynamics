#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.special.erf as erf
import scipy.constants as const


class DifferentialArea:
    """
    Require position, normal, angular_vel be 3-d vector, and normal should 
    be unit vector

    Parameters:
    ===========
    position, angular_vel: 3-d vector (double)
    normal (inner), binormal, tangent: 3-d unit vector (double), local frame

    """
    def __init__(self, position=None, angular_vel=None, normal=None):
	    if position is None:
            self.position = np.array([-1.0, 0.0, 0.0])
        else: 
            self.position = position
        if angular_vel is None:
            self.angular_vel = np.array([0.0, 0.0, 1.0])
        else:
            self.angular_vel = angular_vel
        if normal is None:
            self.normal = -self.position/np.linalg.norm(self.position)
        else:
            self.normal = normal/np.linalg.norm(normal)
        
        self.velocity = np.cross(angular_vel, position)
        assert(self.velocity != np.zeros([3])
        self.binormal = np.cross(self.tangent, self.normal)
        self.tangent = self.velocity / np.linalg.norm(self.velocity)

    def maxwellian_vdf(self, flow, velocity):
        """
        Maxwellian velocity distribution function of the molecules w.r.r to the
        differential area.

        Parameters:
        ===========
        flow: `~FreeMolecularFlow`
        velocity: 3-d vector, m/s 

        Return:
        =======
        vdf: double
        """
        t = flow.temperature
        v = flow.velocity - self.velocity
        vdf = (2.0*np.pi*const.R*t)**(-1.5)*np.exp(
            -0.5/const.R/t*np.dot(velocity-v, velocity-v)
        )
        return vdf

    def pressure(self, flow, wall_temperature=None, sigma=None):
        """
        Pressure Pn and tangential stresses Pb, Pt.

        Parameter:
        ==========
        flow: `~FreeMolecularFlow`
        wall_temperature: optional, float, unit = Kelvin
        sigma: optional, array-like, len(sigma)=2
            Normal accommodation parameter sigma_n, and tangent 
        accommodation parameter sigma_t

        Return:
        =======
        [Pni, Pbi, Pti]: Ndarray, size 3
        """
        tw = 300.0 if wall_temperature is None else wall_temperature
        if sigma is None:
            sigma_t, sigma_n = 1.0, 1.0
        else:
            assert(len(sigma)==2)
            sigma_n, sigma_t = sigma

        scale = np.sqrt(2.0*const.R*flow.temperature)
        s_flow = flow.velocity / scale
        s_area = s_flow - self.velocity / scale
        w_area = self.angular_vel / scale
        a = flow.density/2.0/np.sqrt(np.pi)*scale**2.0
        
        frame = np.matrix([self.normal, self.binormal, self.tangent]).transpose()
        sn, sb, st = np.array(np.dot(s_area, frame)).flatten()

        def _chi(x):
            return np.exp(-x**2) + x * np.sqrt(np.pi) * (1.0+erf(x))
        def _xi(x):
            return _chi(x)*x + 0.5 * np.sqrt(np.pi) * (1.0+erf(x))
        
        Pni = a * _xi(sn)
        Pbi = a * sb * _chi(sn)
        Pti = a * st * _chi(sn)
        Pw = a * np.sqrt(np.pi)/2.0*np.sqrt(tw/flow.temperature)*_chi(sn)
        
        Pn = (2.0 - sigma_n) * Pni + sigma_n * Pw
        Pb = sigma_t * Pbi
        Pt = sigma_t * Pti

        return np.array([Pn, Pb, Pt])

class FreeMolecularFlow:
    """
    Define a free molecular flow

    Parameters:
    ===========
    temperature: kelvins, double
    density: kg / m^3, double
    velocity: m/s, 3-d vector, double

    """
    def __init__(self, temperature=None, density=None, velocity=None):
        self.temperature = 1000.0 if temperature == None else temperature
        self.density = 1.0e-12 if density == None else density
        self.velocity = 7500.0 if velocity == None else velocity
            

def test():
    diff_area = DifferentialArea()


if __name__ == '__main__':
    
    test()
