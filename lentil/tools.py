# -*- coding: utf-8 -*-
# Copyright 2013-2017 Nate Bogdanowicz

import numpy as np
from functools import reduce
from numpy import sqrt, complex, sign, linspace, pi
from scipy.special import erfinv
from .elements import OpticalElement, Identity
from . import Q_


def _get_real(q):
    """Temporary hack to add units to `real`"""
    return Q_(q.magnitude.real, q.units)


def _get_imag(q):
    """Temporary hack to add units to `real`"""
    return Q_(q.magnitude.imag, q.units)


class BeamParam(object):
    """An object representing a reduced complex beam parameter

    Uses Siegman's convention of q-hat, i.e. q is 'reduced' by n.
    """
    def __init__(self, wavlen, zR, z0, n=1):
        """
        Parameters
        ----------
        wavlen : Quantity([Length])
          The wavelength of the beam in this medium
        zR : Quantity([Length])
          The Rayleigh range
        z0 : Quantity([Length])
          The z-position of the focus
        n : int or float
          The refractive index where the beam parameter is being defined
        """
        self.wavlen = Q_(wavlen).to('nm')
        z0 = Q_(z0).to('mm')
        zR = Q_(zR).to('mm')
        self.q0 = -z0 + 1j * zR  # q(z=0)

    @property
    def zR(self):
        """Rayleigh range"""
        return _get_imag(self.q0)

    @property
    def z0(self):
        """The z-position of the focus"""
        return -_get_real(self.q0)

    @property
    def w0(self):
        """Waist size of the beam"""
        return self.waist()

    def waist(self, n=1):
        """Waist size of the beam. Allows you to specify refractive index n."""
        return sqrt((self.wavlen * self.zR) / (n * pi))

    def q(self, z):
        """Value of q at z"""
        return self.q0 + z

    def profile(self, z, n=1, clipping=None):
        z = Q_(z).to('mm')

        scale = 1
        if clipping is not None:
            scale = -erfinv(2*clipping - 1)/sqrt(2)

        return scale * self.waist(n) * sqrt(1 + ((z-self.z0)/self.zR)**2)

    def roc(self, z):
        z = Q_(z).to('mm')

        # R is commonly infinite, so we must ignore divide by zero warnings here
        old_settings = np.seterr(divide='ignore')

        R = 1 / _get_real(1 / self.q(z))

        # Restore divide-by-zero warnings
        np.seterr(**old_settings)

        return R

    def apply_ABCD(self, M, z, wavlen=None):
        """Return a new BeamParam resulting from applying an ABCD matrix M."""
        z = Q_(z).to('mm')
        wavlen = wavlen or self.wavlen
        A, B, C, D = M.elems()
        q = self.q(z)
        q_new = (A*q + B) / (C*q + D)
        return BeamParam(wavlen, z0=(z-_get_real(q_new)), zR=_get_imag(q_new))


def _find_cavity_mode(M):
    """
    Returns 1/q for a cavity eigenmode given the effective cavity matrix M.
    """
    A, B, C, D = M.elems()

    # From Siegman: Lasers, Chapter 21.1
    term1 = (D-A)/(2*B)
    term2 = 1/B*sqrt(complex(((A+D)/2)**2 - 1))
    sgn = sign(_get_imag(term2))

    # Choose transversely confined solution
    q_r = term1 - sgn*term2

    # Check stability to perturbation
    if ((A+D)/2)**2 > 1:
        raise Exception('Resonator is unstable')
    return q_r


def find_cavity_modes(elems):
    """
    Find the eigenmodes of an optical cavity.

    Parameters
    ----------
    elems : list of OpticalElements
        ordered list of the cavity elements

    Returns
    -------
    qt_r, qs_r : complex Quantity objects
        1/q for the tangential and sagittal modes, respectively. Has
        units of 1/[length].
    """
    qt_r = _find_cavity_mode(reduce(lambda x, y: y*x, [el.tan for el in elems]))
    qs_r = _find_cavity_mode(reduce(lambda x, y: y*x, [el.sag for el in elems]))
    return qt_r, qs_r


def _unitful_linspace(start, stop, *args, **kwds):
    start, stop = Q_(start), Q_(stop)
    units = start.units
    raw_pts = linspace(start.m, stop.m_as(units), *args, **kwds)
    return Q_(raw_pts, units)


def get_profiles(q, orientation, elements, z_start=None, z_end=None, n=1, clipping=None):
    zs, profiles, RoCs = [], [], []

    elems = []
    ns = []  # n2(i) for each element el(i)
    for el in elements:
        if isinstance(el, OpticalElement):
            elems.append(el)
            ns.append(n)
        else:
            ns[-1] = el

    # elems = sorted(elements, key=lambda el: el.z)
    if z_start is not None:
        elems.insert(0, Identity(z_start))
        ns.insert(0, n)
    if z_end is not None:
        elems.append(Identity(z_end))
        ns.append(n)

    print(elems)
    print(ns)

    el = elems.pop(0)
    n1 = n
    for next_el, n2 in zip(elems, ns):
        M = (el.sag if orientation == 'sagittal' else el.tan)(n1, n2)
        q = q.apply_ABCD(M, el.z)

        z = _unitful_linspace(el.z, next_el.z, 1000)
        zs.append(z)
        profiles.append(q.profile(z, el.n, clipping))
        RoCs.append(q.roc(z))
        el = next_el
        n1 = n2

    return zs, profiles, RoCs
