# -*- coding: utf-8 -*-
# Copyright 2013-2018 Nate Bogdanowicz
from numbers import Number
from numpy import cos, sin, arcsin
from .util import Q_, pairwise


def _parse_angle(ang):
    """
    If ang is a number, treats it as in degrees. Otherwise it does the usual
    unit parsings from Q_
    """
    if isinstance(ang, Number):
        ang = Q_(ang, 'deg')
    else:
        ang = Q_(ang)
    return ang


def ensure_units(value, units):
    if value == 0:
        return Q_(0, units)  # Allow passing in bare zeros
    return Q_(value).to(units)


class ABCD(object):
    """A simple ABCD (ray transfer) matrix class.

    ABCD objects support mutiplication with scalar numbers and other ABCD
    objects.
    """
    def __init__(self, A, B, C, D):
        """Create an ABCD matrix from its elements.

        The matrix is a 2x2 of the form::

            [A B]
            [C D]

        Each matrix should be defined using Siegman's convention.

        Parameters
        ----------
        A,B,C,D : Quantity objects
            `A` and `D` are dimensionless. `B` has units of [length] (e.g. 'mm'
            or 'rad/mm'), and `C` has units of 1/[length].
        """
        self.A = ensure_units(A, 'dimensionless')
        self.B = ensure_units(B, 'mm/rad')
        self.C = ensure_units(C, 'rad/mm')
        self.D = ensure_units(D, 'dimensionless')

    def __mul__(self, other):
        if isinstance(other, ABCD):
            A = self.A*other.A + self.B*other.C
            B = self.A*other.B + self.B*other.D
            C = self.C*other.A + self.D*other.C
            D = self.C*other.B + self.D*other.D
            return ABCD(A, B, C, D)
        elif isinstance(other, (int, float)):
            return ABCD(A*other, B*other, C*other, D*other)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, ABCD):
            A = other.A*self.A + other.B*self.C
            B = other.A*self.B + other.B*self.D
            C = other.C*self.A + other.D*self.C
            D = other.C*self.B + other.D*self.D
            return ABCD(A, B, C, D)
        elif isinstance(other, (int, float)):
            return ABCD(self.A*other, self.B*other, self.C*other, self.D*other)
        return NotImplemented

    def _stringify(self, quantity):
        s = '{:.2f}'.format(quantity)
        s = s.replace('dimensionless', '')
        s = s.replace('millimeter', 'mm')
        s = s.replace('radian', 'rad')
        s = s.replace(' / ', '/')
        return s.strip()

    def __repr__(self):
        strA = self._stringify(self.A)
        strB = self._stringify(self.B)
        strC = self._stringify(self.C)
        strD = self._stringify(self.D)

        diff = len(strA) - len(strC)
        if diff > 0:
            strC = " "*(diff-diff//2) + strC + " "*(diff//2)
        else:
            diff = -diff
            strA = " "*(diff-diff//2) + strA + " "*(diff//2)

        diff = len(strB) - len(strD)
        if diff > 0:
            strD = " "*(diff-diff//2) + strD + " "*(diff//2)
        else:
            diff = -diff
            strB = " "*(diff-diff//2) + strB + " "*(diff//2)

        return "[{} , {}]\n[{} , {}]".format(strA, strB, strC, strD)

    def elems(self):
        """Get the matrix elements.

        Returns
        -------
        A, B, C, D : tuple of Quantity objects
            The matrix elements
        """
        return self.A, self.B, self.C, self.D


class OpticalElement(object):
    def __init__(self, z):
        self.n = 1
        self.z = ensure_units(z, 'mm')

        # For compound elements
        self.dz = Q_(0, 'mm')
        self.elems = [self]
        self.ns = []

    @property
    def z_left(self):
        return self.elems[0].z

    @property
    def z_right(self):
        return self.elems[-1].z

    def __mul__(self, other):
        tan = self.tan * other.tan
        sag = self.sag * other.sag
        return OpticalElement(tan, sag)

    def __rmul__(self, other):
        tan = other.tan * self.tan
        sag = other.sag * self.sag
        return OpticalElement(tan, sag)

    def __repr__(self):
        return "<{} z={:~.1f}>".format(self.__class__.__name__, self.z)

    def tan(self, n1, n2):
        """The ABCD matrix for a tangential beam, given left and right refractive indices"""
        return self._tan(n1, n2) if callable(self._tan) else self._tan

    def sag(self, n1, n2):
        """The ABCD matrix for a sagittal beam, given left and right refractive indices"""
        return self._sag(n1, n2) if callable(self._sag) else self._sag

    def path_info(self, n1, n2):
        """list of tuples (zs, M_tan, M_sag, (n1,n2)) for each internal ABCD matrix"""
        return list(self._gen_path_info(n1, n2))

    def _gen_path_info(self, n1, n2):
        n_pairs = pairwise([n1] + self.ns + [n2])
        for el, (na, nb) in zip(self.elems, n_pairs):
            yield (el.z, el.tan(na,nb), el.sag(na,nb), (na, nb))


class FlatSlab(OpticalElement):
    def __init__(self, z, d, n, aot=None):
        """
        Parameters
        ----------
        d : Quantity([Length])
          Thickness of the slab, measured normal to the flats. If `aot` is nonzero, the optical path
          length will be greater than this.
        """
        OpticalElement.__init__(self, z)
        d = ensure_units(d, 'mm')
        if aot:
            aot = _parse_angle(aot)
            self.dz = d / cos(aot)
        else:
            self.dz = d

        self.left = Interface(self.z, None, aot=aot)
        self.right = Interface(self.z+self.dz, None, aot=aot)
        self.elems = [self.left, self.right]
        self.ns = [n]


class Identity(OpticalElement):
    """A dummy element"""
    def __init__(self, z):
        OpticalElement.__init__(self, z)
        self.n = 1
        self.z = ensure_units(z, 'mm')
        self._tan = self._sag = IdentityABCD()


class IdentityABCD(ABCD):
    def __init__(self):
        ABCD.__init__(self, 1, 0, 0, 1)


class SpaceABCD(ABCD):
    # FIXME: Should we not divide by n because of our BeamParam convention?
    def __init__(self, d, n=1):
        d = ensure_units(d, 'mm/rad')
        ABCD.__init__(self, 1, d/n, 0, 1)


class Lens(OpticalElement):
    """A thin lens"""
    def __init__(self, z, f):
        """
        Parameters
        ----------
        f : Quantity or str
            The focal length of the lens
        """
        OpticalElement.__init__(self, z)
        self._tan = self._sag = LensABCD(f)


class LensABCD(ABCD):
    def __init__(self, f):
        f = ensure_units(f, 'mm/rad')
        ABCD.__init__(self, 1, 0, -1/f, 1)


class CylindricalLens(OpticalElement):
    """A thin cylindrical lens"""
    def __init__(self, z, f, orientation):
        OpticalElement.__init__(self, z)
        if orientation == 'tangential':
            self._tan = LensABCD(f)
            self._sag = IdentityABCD()
        elif orientation == 'sagittal':
            self._tan = IdentityABCD()
            self._sag = LensABCD(f)
        else:
            raise ValueError("orientation must be 'tangential' or 'sagittal'")


class Mirror(OpticalElement):
    """A mirror, possibly curved"""
    def __init__(self, z, R=None, aoi=0):
        """
        Parameters
        ----------
        R : Quantity or str, optional
            The radius of curvature of the mirror's spherical surface. Defaults
            to `None`, indicating a flat mirror.
        aoi : Quantity or str or number, optional
            The angle of incidence of the beam on the mirror, defined as the
            angle between the mirror's surface normal and the beam's axis.
            Defaults to 0, indicating normal incidence.
        """
        OpticalElement.__init__(self, z)
        R = Q_(R).to('mm') if R else Q_(float('inf'), 'mm')
        aoi = _parse_angle(aoi)

        self._tan = MirrorABCD(R, aoi, 'tangential')
        self._sag = MirrorABCD(R, aoi, 'sagittal')


class MirrorABCD(ABCD):
    def __init__(self, R, aoi, orientation):
        if orientation == 'tangential':
            Re = R * cos(aoi)
        else:
            Re = R / cos(aoi)

        ABCD.__init__(self, 1, 0, -2/Re, 1)


class Interface(OpticalElement):
    """An interface between media with different refractive indices"""
    def __init__(self, z, R=None, aoi=None, aot=None):
        """
        Parameters
        ----------
        R : Quantity or str, optional
            The radius of curvature of the interface's spherical surface, in
            units of length. Defaults to `None`, indicating a flat interface.
        aoi : Quantity or str or number, optional
            The angle of incidence of the beam relative to the interface,
            defined as the angle between the interface's surface normal and the
            _incident_ beam's axis.  If not specified but `aot` is given, aot
            will be used. Otherwise, `aoi` is assumed to be 0, indicating
            normal incidence. A raw number is assumed to be in units of
            degrees.
        aot : Quantity or str or number, optional
            The angle of transmission of the beam relative to the interface,
            defined as the angle between the interface's surface normal and
            the *transmitted* beam's axis. See `aoi` for more details.
        """
        OpticalElement.__init__(self, z)

        if aoi is None:
            if aot is None:
                a1 = Q_(0)
                a2 = Q_(0)
            else:
                a2 = _parse_angle(aot)
                a1 = None  # Calculate when given n's
        else:
            if aot is None:
                a1 = _parse_angle(aoi)
                a2 = None  # Calculate when given n's
            else:
                raise Exception("Cannot specify both aoi and aot")

        self._tan = lambda n1,n2: InterfaceABCD(n1, n2, R, a1, a2, 'tangential')
        self._sag = lambda n1,n2: InterfaceABCD(n1, n2, R, a1, a2, 'sagittal')


class InterfaceABCD(ABCD):
    def __init__(self, n1, n2, R, a1, a2, orientation):
        R = ensure_units(R or Q_(float('inf'), 'mm'), 'mm')
        if a1 is None:
            a1 = arcsin(n2/n1*sin(a2))
        if a2 is None:
            a2 = arcsin(n1/n2*sin(a1))
        if orientation == 'tangential':
            dne = (n2 * cos(a2) - n1 * cos(a1)) / (cos(a1) * cos(a2))
            ABCD.__init__(self, cos(a2)/cos(a1), 0, dne/R, cos(a1)/cos(a2))
        else:
            dne = n2 * cos(a2) - n1 * cos(a1)
            ABCD.__init__(self, 1, 0, dne/R, 1)
