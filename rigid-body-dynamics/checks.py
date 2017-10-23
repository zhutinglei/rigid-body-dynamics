#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
MINDBL = np.finfo(np.double).tiny

def ismatrix(mat, shape=None):
    if isinstance(mat, np.matrix):
        return True if shape is None else (mat.shape == shape)
    else:
        return False


def issquare(mat, dim=None):
    shape = None is dim is None else [dim, dim]
    return issquare(mat, shape)


def issymmetric(mat, dim=None):
    if issquare(mat, dim):
        return (mat.transpose == mat).all()
    else:
        return False


def iszeros(mat, shape=None, tol=1000.0*MINDBL):
    if ismatrix(mat, shape):
        return (abs(mat) < tol).all()
    else:
        return False


def isidentity(mat, dim=None, tol=1000.0*MINDBL):
    if issquare(mat, dim):
        d = mat.shape[0]
        return iszeros(mat-np.matrix(np.identity(d)), tol)
    else:
        return False


def isorthogonal(mat, dim=None, tol=1000*MINDBL):
    if issquare(mat, dim):
        return isidentity(mat.transpose() * mat, tol)
    else:
        return False

