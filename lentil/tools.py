# -*- coding: utf-8 -*-
# Copyright 2013-2018 Nate Bogdanowicz
import numpy as np
from itertools import chain
from collections import deque
from functools import reduce
from numpy import sqrt, complex, pi, exp
from scipy.special import erfinv
from scipy.optimize import minimize
from .elements import OpticalElement, Identity, SpaceABCD, ensure_units
from .util import u, Q_, pairwise, unitful_linspace, argrelmin


def modsq(z):
    return z.real**2 + z.imag**2


def _get_real(q):
    """Temporary hack to add units to `real`"""
    return Q_(q.magnitude.real, q.units)


def _get_imag(q):
    """Temporary hack to add units to `real`"""
    return Q_(q.magnitude.imag, q.units)


class BeamParam(object):
    """An object representing a reduced complex beam parameter

    Uses Siegman's convention of q-tilde, i.e. q is *not* 'reduced' by n (pp. 784).
    """
    def __init__(self, lambda0, zR, z0, n=1):
        """
        Parameters
        ----------
        lambda0 : Quantity([Length])
          The vacuum wavelength of the beam
        zR : Quantity([Length])
          The Rayleigh range
        z0 : Quantity([Length])
          The z-position of the focus
        n : int or float
          The refractive index where the beam parameter is being defined
        """
        self.lambda0 = Q_(lambda0).to('nm')
        self.n = n
        z0 = Q_(z0).to('mm')
        zR = Q_(zR).to('mm')
        self.q0 = -z0 + 1j * zR  # q(z=0)

    def __repr__(self):
        return ('<BeamParam w0={:~.3f}, z0={:~.3f}, n={:.3f}>'
                ''.format(self.w0, self.z0, float(self.n)))

    @classmethod
    def from_q(cls, q, lambda0, z='0mm', n=1):
        """
        Parameters
        ----------
        q : Quantity([Length])
          The complex beam parameter.
        """
        z = Q_(z).to('mm')
        zR = _get_imag(q)
        z0 = z - _get_real(q)
        return BeamParam(lambda0, zR, z0, n)

    @classmethod
    def from_waist(cls, lambda0, w0, z0='0mm', n=1):
        """
        Parameters
        ----------
        lambda0 : Quantity([Length])
          The vacuum wavelength of the beam
        w0 : Quantity([Length])
          The waist (radius) of the beam *in the medium given by `n`*
        z0 : Quantity([Length])
          The z-position of the focus
        n : int or float
          The refractive index where the beam parameter is being defined
        """
        lambda0 = Q_(lambda0).to('nm')
        w0 = Q_(w0).to('mm')
        z0 = Q_(z0).to('mm')
        zR = pi * w0**2 * n / lambda0
        return BeamParam(lambda0, zR, z0, n)

    @classmethod
    def from_widths(cls, lambda0, z1, w1, z2, w2, n=1, focus_location='infer'):
        """
        Parameters
        ----------
        lambda0 : Quantity([Length])
          The vacuum wavelength of the beam
        z1, z2 : Quantity([Length])
          The z-position of each measured width.
        w1, w2 : Quantity([Length])
          The measured beam widths (radii)
        n : int or float
          The refractive index in which both measurements were made.
        focus_location : str
          One of 'infer', 'left', 'right', and 'between'
        """
        lambda0 = Q_(lambda0).to('nm')
        z1 = Q_(z1).to('mm')
        z2 = Q_(z2).to('mm')
        w1 = Q_(w1).to('mm')
        w2 = Q_(w2).to('mm')

        if z1 < z2:
            z_left, z_right = z1, z2
            w_left, w_right = w1, w2
        else:
            z_left, z_right = z2, z1
            w_left, w_right = w2, w1

        if focus_location == 'infer':
            focus_location = 'left' if w_left <= w_right else 'right'

        if focus_location == 'left':
            if w_left > w_right:
                raise ValueError('Focus cannot be to the left; left spot is larger than right spot')
            zR_sign = +1
            z0_sign = -1
        elif focus_location == 'right':
            if w_right > w_left:
                raise ValueError('Focus cannot be to the right; right spot is larger than left '
                                 'spot')
            zR_sign = +1
            z0_sign = +1
        elif focus_location == 'between':
            zR_sign = -1
            z0_sign = +1 if z1 < z2 else -1
        else:
            raise ValueError
        # NOTE: wavlen should be wavlen in medium
        # zR expression derived using Wolfram Cloud
        wavlen = lambda0 / n
        zR_num = (w1**2 + w2**2) + zR_sign * 2*sqrt(w1**2*w2**2 - ((wavlen/pi*(z1-z2))**2))
        zR_den = pi/wavlen*((w1**2-w2**2)/(z1-z2))**2 + 4*wavlen/pi
        zR = zR_num/zR_den
        z0 = z1 + z0_sign * sqrt(zR * (pi/wavlen*w1**2 - zR))
        return cls(lambda0, zR, z0, n=n)

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
        """Waist size (radius) of the beam"""
        return sqrt((self.lambda0 * self.zR) / (self.n * pi)).to('mm')

    def q(self, z):
        """Value of the complex number q when evaluated at z"""
        z = ensure_units(z, 'mm')
        return self.q0 + z

    def profile(self, z, clipping=None):
        z = ensure_units(z, 'mm')

        scale = 1
        if clipping is not None:
            scale = -erfinv(2*clipping - 1)/sqrt(2)

        return scale * self.w0 * sqrt(1 + ((z-self.z0)/self.zR)**2)

    def roc(self, z):
        z = Q_(z).to('mm')

        # R is commonly infinite, so we must ignore divide by zero warnings here
        old_settings = np.seterr(divide='ignore')

        R = 1 / _get_real(1 / self.q(z))

        # Restore divide-by-zero warnings
        np.seterr(**old_settings)

        return R

    def u(self, r, z):
        """Normalized amplitude u(r,z), without axial phase term.

        Because of the missing axial phase term, the phase at r=0 is zero.
        """
        wz = self.profile(z)
        axial_amp = 1 / wz
        radial_amp = exp(-(r/wz)**2)
        radial_phase = exp(-1j*pi*r**2 * self.n/(self.lambda0*self.roc(z)))
        return sqrt(2/pi) * axial_amp * radial_amp * radial_phase

    def mode_overlap(self, other, z, check=True):
        """The integrated mode overlap of two beams at a given z position"""
        if check:
            if self.lambda0 != other.lambda0:
                raise ValueError('BeamParam wavelengths differ: {} != {}'.format(self.lambda0,
                                                                                 other.lambda0))
            if self.n != other.n:
                raise ValueError("BeamParam n's differ: {} != {}".format(self.n, other.n))

        z = ensure_units(z, 'mm')
        w0 = min(self.w0, other.w0)  # Most overlap fits in smaller beam
        r = unitful_linspace('0mm', 10*w0, 1000)
        u1 = self.u(r, z)
        u2 = other.u(r, z)
        # u's are normalized, so we don't have to divide by their self-overlaps
        integrand = (2*pi*r * np.conj(u1) * u2).m_as('1/mm')
        r_mm = r.m_as('mm')
        return modsq(np.trapz(integrand, r_mm))

    def apply_ABCD(self, M, z, n2=None):
        """Return a new BeamParam resulting from applying an ABCD matrix M."""
        z = Q_(z).to('mm')
        n1 = self.n
        n2 = n2 or n1
        lambda2 = self.lambda0 / n2
        A, B, C, D = M.elems()
        q1 = self.q(z)
        q2 = n2 * (A*q1/n1 + B) / (C*q1/n1 + D)
        return BeamParam(lambda2, z0=(z-_get_real(q2)), zR=_get_imag(q2), n=n2)

    def unapply_ABCD(self, M, z, n1=None):
        # FIXME: add in factors of n1, n2
        z = Q_(z).to('mm')
        n2 = self.n
        n1 = n1 or n2
        A, B, C, D = M.elems()
        q2 = self.q(z)
        q1 = n1 * (D*q2/n2 - B) / (-C*q2/n2 + A)
        return BeamParam.from_q(q1, z=z, lambda0=self.lambda0, n=n1)

    def apply_optical_elements(self, elems, tangential=True, n=1):
        """Make a new BeamParam by passing this BeamParam through a list of optical elements"""
        elems, ns = extract_elems_ns(elems, n)
        zs, Ms_tan, Ms_sag, n_pairs = path_info(elems, ns)
        Ms = Ms_tan if tangential else Ms_sag
        q = self
        for M, z_M, (n1, n2) in zip(Ms, zs, n_pairs):
            q = q.apply_ABCD(M, z_M, n2=n2)

        return q


def mode_match(qa, qb, z, lenses):
    """Calculate optimized mode matching using the given lenses"""
    z_test = ensure_units(z, 'mm')
    lenses = lenses[:]
    def cost_func(zs):
        for lens, z in zip(lenses, zs):
            lens.z = Q_(z, 'mm')
        q = qa.apply_optical_elements(lenses)
        return -q.mode_overlap(qb, z_test)
    z0 = [lens.z.m_as('mm') for lens in lenses]
    result = minimize(cost_func, z0)

    if not result.success:
        raise ValueError('Minimization failed: {}'.format(result.message))

    return lenses


def _find_cavity_mode(M):
    """
    Returns 1/q for a cavity eigenmode given the effective cavity matrix M.
    """
    A, B, C, D = M.elems()

    # From Siegman: Lasers, Chapter 21.1
    term1 = (D-A)/(2*B)
    term2 = 1/B*sqrt(complex(((A+D)/2)**2 - 1))

    # Choose transversely confined solution
    q_r_a = term1 - term2
    q_r_b = term1 + term2

    if _get_imag(q_r_a).m < 0:
        q_r = q_r_a
    elif _get_imag(q_r_b).m < 0:
        q_r = q_r_b
    else:
        raise Exception('No confined solution')

    # Check stability to perturbation
    if ((A+D)/2)**2 > 1:
        raise Exception('Resonator is unstable')
    return q_r


def find_cavity_modes(elems, lambda0, n=1):
    """
    Find the eigenmodes of an optical cavity.

    Parameters
    ----------
    elems : list of OpticalElements
        ordered list of the cavity elements

    Returns
    -------
    qt, qs : BeamParams
        beam parameter for the tangential and sagittal modes, respectively.
    """
    lambda0 = Q_(lambda0).to('nm')
    tans, sags = [], []
    elems, ns = _get_element_indices(elems, n)

    el, n1 = elems.pop(0), n
    for next_el, n2 in zip(elems, ns):
        sags.append(el.sag(n1, n2))
        tans.append(el.tan(n1, n2))
        el, n1 = next_el, n2

    qt_r = _find_cavity_mode(reduce(lambda x, y: y*x, tans))
    qs_r = _find_cavity_mode(reduce(lambda x, y: y*x, sags))

    qt = BeamParam.from_q(1/qt_r, lambda0=lambda0, n=n)
    qs = BeamParam.from_q(1/qs_r, lambda0=lambda0, n=n)
    return qt, qs


def _get_element_indices(elements, n):
    elems = []
    ns = []  # n2(i) for each element el(i)
    for el in elements:
        if isinstance(el, OpticalElement):
            elems.append(el)
            ns.append(n)
        else:
            ns[-1] = el

    return elems, ns


def get_profiles(q, orientation, elements, z_start=None, z_end=None, n=1, clipping=None):
    zs, profiles, RoCs = [], [], []
    elems, ns = _get_element_indices(elements, n)
    if z_start is not None:
        elems.insert(0, Identity(z_start))
        ns.insert(0, n)
    if z_end is not None:
        elems.append(Identity(z_end))
        ns.append(n)

    zs, Ms, n_pairs = [], [], []
    for (el, next_el), (n1,n2) in zip(pairwise(elems), pairwise(ns)):
        # Transform beam parameter
        # FIXME: Add sag_list()
        el_zs, el_Ms, el_n_pairs = el.tan_list(n1, n2)
        #M = (el.sag if orientation == 'sagittal' else el.tan)(n1, n2)

        for z_M, M, (n1_M, n2_M) in zip(el_zs, el_Ms, el_n_pairs):
            q = q.apply_ABCD(M, z_M, n2=el.n)

            # Calculate beam info within region
            z = unitful_linspace(el.z, next_el.z, 1000)
            zs.append(z)
            profiles.append(q.profile(z, clipping))
            RoCs.append(q.roc(z))

    return zs, profiles, RoCs


def listify(gen_func):
    def func(*args, **kwds):
        return list(gen_func(*args, **kwds))
    return func


def extract_elems_ns(elems_and_ns, n=1.):
    elems = []
    ns = [n]
    z = None
    for item in elems_and_ns:
        if isinstance(item, OpticalElement):
            if z is not None and item.z < z:
                raise ValueError('Elements are not z-sorted')
            z = item.z
            elems.append(item)
            ns.append(n)
        else:
            # Update previous n
            n = float(item)
            ns[-1] = n
    return elems, ns


class BeamPath(object):
    def __init__(self, elements):
        """A beam path to propagate through.

        Parameters
        ----------
        elements : list
             An ordered list of optical elements, optionally separated by numbers indicating the
        refractive indices. If a refractive index is omitted, it is assumed to be the same as the
        previous index. If not given, the initial index is 1.
        """
        self.elems, self.ns = extract_elems_ns(elements)

    def __repr__(self):
        elem_strs = ('  ' + repr(e) for e in self.elems)
        n_strs = ('    ({})'.format(n) for n in self.ns)
        return '<BeamPath [\n{}\n]>'.format('\n'.join(zipper(n_strs, elem_strs)))

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError('Index must be a slice')
        if key.step is not None:
            raise ValueError('Index slice cannot have a step')
        start, stop, _ = key.indices(len(self.elems))
        path = BeamPath([])
        path.elems = self.elems[start:stop]
        path.ns = self.ns[start:stop+1]
        return path

    @property
    def n_pairs(self):
        return list(pairwise(self.ns))

    @property
    @listify
    def elems_with_spaces(self):
        yield self.elems[0]
        z_prev = self.elems[0].z
        for el, n in zip(self.elems[1:], self.ns[1:]):
            #yield Space(z_prev, d=el.z-z_prev, n=n)
            yield el
            z_prev = el.z + el.dz

    @property
    @listify
    def ns_with_spaces(self):
        yield self.ns[0]
        for n in self.ns[1:]:
            yield n  # for Space
            yield n  # for Element

    @property
    def n_pairs_with_spaces(self):
        return list(pairwise(self.ns_with_spaces))

    @property
    def Ms_tan(self):
        #yield from self.elems[0].tan_list(self.)
        for el, (n1,n2) in zip(self.elems_with_spaces, self.n_pairs_with_spaces):
            pass

    def get_profiles(self, q, orientation, z_start=None, z_end=None, n=1, z_q=None, clipping=None):
        zs, Ms_tan, Ms_sag, n_pairs = path_info(self.elems, self.ns, z_start, z_end, add_spaces=False)
        Ms = Ms_tan if orientation == 'tangential' else Ms_sag

        # TODO: Maybe make this more efficient
        if z_q is not None:
            # Propagate q back to start
            z_q = ensure_units(z_q, 'mm')
            for M, z_M, (n1, n2) in reversed(list(zip(Ms, zs, n_pairs))):
                if z_M > z_q:
                    continue  # Apply only M's left of z_q
                q = q.unapply_ABCD(M, z_M, n1=n1)

        all_zs, profiles, RoCs = [], [], []
        for M, (z_M, z_next_M), (n1, n2) in zip(Ms, pairwise(zs), n_pairs):
            # FIXME: Clear up this wavlen/n business
            q = q.apply_ABCD(M, z_M, n2=n2)
            z = unitful_linspace(z_M, z_next_M, 1000)
            all_zs.append(z)
            profiles.append(q.profile(z, clipping))
            RoCs.append(q.roc(z))
        return all_zs, profiles, RoCs

    def find_cavity_modes(self, lambda0, n=1, ring=True):
        """Find the eigenmodes of an optical cavity defined by this BeamPath

        The first and last elements of this BeamPath should be mirrors.

        Parameters
        ----------
        lambda0 : Quantity([Length])
          The vacuum wavelength of the beam
        ring : bool
            If True, solve as a ring cavity. Otherwise, assumes the beam travels back through the
            optical elements, in a standing-wave configuration (like a Fabry-Perot cavity).

        Returns
        -------
        qt, qs : BeamParams
            beam parameter for the tangential and sagittal modes, respectively.
        """
        # FIXME: Verify we're doing the right thing regarding wavelengths and n's
        lambda0 = Q_(lambda0).to('nm')
        zs, Ms_tan, Ms_sag, n_pairs = path_info(self.elems, self.ns, add_spaces=True)

        if not ring:
            Ms_tan = Ms_tan + Ms_tan[-2:0:-1]  # ABCDE -> ABCDEDCB
            Ms_sag = Ms_sag + Ms_sag[-2:0:-1]  # ABCDE -> ABCDEDCB

        qt_r = _find_cavity_mode(combine(Ms_tan))
        qs_r = _find_cavity_mode(combine(Ms_sag))

        qt = BeamParam.from_q(1/qt_r, lambda0=lambda0, n=n, z=zs[0])
        qs = BeamParam.from_q(1/qs_r, lambda0=lambda0, n=n, z=zs[0])
        return qt, qs

    def plot_profile(self, q_start_t, q_start_s, z_q=None, z_start=None, z_end=None, cyclical=False,
                     names=tuple(), clipping=None, show_axis=False, show_waists=False, zeroat=0,
                     zunits='mm', runits='um', ax=None, **kwds):
        zs, profs_t, RoCs = self.get_profiles(q_start_t, 'tangential', z_start, z_end, 1, z_q=z_q,
                                              clipping=clipping)
        zs, profs_s, RoCs = self.get_profiles(q_start_s, 'sagittal', z_start, z_end, 1, z_q=z_q,
                                              clipping=clipping)
        # Convert lists of Quantity-arrays
        from .plotting import _magify
        zs_mag = _magify(zs, zunits)
        profs_t_mag = _magify(profs_t, runits)
        profs_s_mag = _magify(profs_s, runits)
        # RoC_mag = _magify(RoC, runits)

        if ax is None:
            from matplotlib.pyplot import subplots
            _, ax = subplots()
        margin = .0002e3
        ax.set_xlim([zs_mag[0][0]-margin, zs_mag[-1][-1]+margin])

        # Concatenate list into a single Quantity-array
        z_mag = np.concatenate(zs_mag)
        prof_t_mag = np.concatenate(profs_t_mag)
        prof_s_mag = np.concatenate(profs_s_mag)

        t_kwds = dict(color='b', linewidth=1.5)
        t_kwds.update(kwds)
        s_kwds = dict(color='r', linewidth=1.5)
        s_kwds.update(kwds)
        ax.plot(z_mag, prof_t_mag, label='Tangential beam', **t_kwds)
        ax.plot(z_mag, prof_s_mag, label='Sagittal beam', **s_kwds)

        if show_waists:
            # Mark waists
            # Should use scipy.signal.argrelextrema, but it's not available before 0.11
            t_waist_indices = argrelmin(prof_t_mag)
            s_waist_indices = argrelmin(prof_s_mag)
            for i in t_waist_indices:
                ax.annotate('{:.3f} {}'.format(prof_t_mag[i], runits),
                            (z_mag[i], prof_t_mag[i]),
                            xytext=(0, 30), textcoords='offset points',
                            ha='center', arrowprops=dict(arrowstyle="->"))
            for i in s_waist_indices:
                ax.annotate('{:.3f} {}'.format(prof_s_mag[i], runits),
                            (z_mag[i], prof_s_mag[i]), xytext=(0, 30),
                            textcoords='offset points', ha='center',
                            arrowprops={'arrowstyle': '->'})

        ax.set_xlabel('Position ({})'.format(zunits))
        if clipping is not None:
            ylabel = ('Distance from beam axis for clipping of ' +
                      '{:.1e} ({})'.format(clipping, runits))
        else:
            ylabel = 'Spot size ({})'.format(runits)
        ax.set_ylabel(ylabel)
        # ax.legend()

        if show_axis:
            ax.set_ylim(bottom=0)
        ax.set_autoscaley_on(False)

        # Pad out names and convert to a list
        names = [names[i] if i < len(names) else '' for i in range(len(zs_mag))]

        if cyclical and names:
            zs_mag.append(zs_mag[-1][-1:])
            names.append(names[0])
            profs_t_mag.append(profs_t_mag[0])
            profs_s_mag.append(profs_s_mag[0])

        # Plot boundary lines and names (if provided)
        for z_mag, prof_t_mag, prof_s_mag, name in zip(zs_mag, profs_t_mag,
                                                       profs_s_mag, names):
            ax.vlines([z_mag[0]], ax.get_ylim()[0], ax.get_ylim()[1],
                      linestyle='dashed', linewidth=2, color=(.5, .5, .5),
                      antialiased=True)

            # Get relevant y boundaries
            ylim0, ylim1 = ax.get_ylim()
            if prof_t_mag[0] < prof_s_mag[0]:
                pmin, pmax = prof_t_mag[0], prof_s_mag[0]
            else:
                pmin, pmax = prof_s_mag[0], prof_t_mag[0]

            region = np.argmax([pmin-ylim0, pmax-pmin, ylim1-pmax])
            if region == 0:
                margin = ylim0
                va = 'bottom'
            elif region == 1:
                margin = pmin + (pmax-pmin)*0.5
                va = 'center'
            else:
                margin = ylim1 - (ylim1-pmax)*0.2
                va = 'top'
            if name:
                ax.text(z_mag[0], margin, name, rotation='vertical', ha='center',
                        va=va, size='xx-large', backgroundcolor='w')


# FIXME: Rename this
def path_info(elems, ns, z_start=None, z_end=None, add_spaces=False):
    """Get all (zs, Ms_tan, Ms_sag, n_pairs) within this path"""
    if len(ns) != len(elems) + 1:
        raise ValueError('Need n+1 indices for n optical elements')

    # Consider *always* adding Identity elements, so we have at least 2 elements
    # Make sure not to modify elems or ns in place
    if z_start is not None:
        elems = [Identity(z_start)] + elems
        ns = [ns[0]] + ns
    if z_end is not None:
        elems = elems + [Identity(z_end)]
        ns = ns + [ns[-1]]

    ## Kinda ugly, but clearer than using itertools
    #lists = ([], [], [], [])
    #for el, (n1, n2) in zip(elems, pairwise(ns)):
    #    for list_, values in zip(lists, el.path_info(n1, n2)):
    #        list_.extend(values)

    info = chain.from_iterable(el.path_info(n1, n2)
                               for el, (n1,n2) in zip(elems, pairwise(ns)))
    if add_spaces:
        old_info, info = info, []
        tup1 = next(old_info)
        info.append(tup1)
        for tup2 in old_info:
            z1, _, _, (n1a, n1b) = tup1
            z2, _, _, (n2a, n2b) = tup2
            M = SpaceABCD(z2-z1, n=n1b)
            info.append((z1, M, M, (n1b, n1b)))
            info.append(tup2)
            tup1 = tup2

    return zip(*info)  # (zs, Ms_tan, Ms_sag, n_pairs)


@listify
def zipper(*seqs):
    """Flatten seqs by intercalating them"""
    iter_queue = deque(iter(seq) for seq in seqs)
    while iter_queue:
        it = iter_queue.popleft()
        try:
            item = next(it)
        except StopIteration:
            pass
        else:
            iter_queue.append(it)
            yield item


def combine(Ms):
    return reduce(lambda x, y: y*x, Ms)


def insertion_index(xs, x):
    """Find the index at which x should be inserted into xs to maintain sort order.

    xs should already be sorted ascending.
    """
    for i, xi in enumerate(xs):
        if xi > x:
            return i
    return i + 1


inf = float('inf')


class Beam(object):
    def __init__(self, path, qx, qy, z_q=Q_(-inf, 'mm'), z_fixed=Q_(-inf, 'mm'), name=None):
        """
        Parameters
        ----------
        z_q : Quantity([Length])
          z-position at which q is defined
        z_fixed : Quantity([Length])
          z-position at which the beam is "fixed". When any optical elements are changed, the beam
          parameter will always be the same at this location.
        """
        z_q = ensure_units(z_q, 'mm')
        z_fixed = ensure_units(z_fixed, 'mm')
        self.path = path
        self.name = name

        zs, Mxs, Mys, n_pairs = path_info(self.elems, self.ns, add_spaces=False)
        i1 = insertion_index(zs, z_q)
        i2 = insertion_index(zs, z_fixed)

        if i1 < i2:
            # Must propagate beam forward to fixed point
            for z_M, Mx, My, (n1, n2) in tuple(zip(zs, Mxs, Mys, n_pairs))[i1:i2]:
                qx = qx and qx.apply_ABCD(Mx, z_M, n2=n2)
                qy = qy and qy.apply_ABCD(My, z_M, n2=n2)
        elif i1 > i2:
            #M Must propagate beam backward to fixed point
            for z_M, Mx, My, (n1, n2) in reversed(tuple(zip(zs, Mxs, Mys, n_pairs))[i2:i1]):
                qx = qx and qx.unapply_ABCD(Mx, z_M, n1=n1)
                qy = qy and qy.unapply_ABCD(My, z_M, n1=n1)

        self._qx0 = qx  # TODO
        self._qy0 = qy  # TODO
        self._z_q = z_q
        self._z_fixed = z_fixed

    @property
    def ns(self):
        return self.path.ns

    @property
    def elems(self):
        return self.path.elems

    @property
    @listify
    def qs(self):
        zs, Mxs, Mys, n_pairs = path_info(self.elems, self.ns)
        qx, qy = self._leftmost_qs(zs, Mxs, Mys, n_pairs)
        yield (qx, qy)
        for z_M, Mx, My, (n1, n2) in zip(zs, Mxs, Mys, n_pairs):
            qx = qx and qx.apply_ABCD(Mx, z_M, n2=n2)
            qy = qy and qy.apply_ABCD(My, z_M, n2=n2)
            yield (qx, qy)

    def _leftmost_qs(self, zs, Mxs, Mys, n_pairs):
        qx, qy = self._qx0, self._qy0

        # Backpropagate if necessary
        i_fixed = insertion_index(zs, self._z_fixed)
        for z_M, Mx, My, (n1, n2) in reversed(tuple(zip(zs, Mxs, Mys, n_pairs))[:i_fixed]):
            qx = qx and qx.unapply_ABCD(Mx, z_M, n1=n1)
            qy = qy and qy.unapply_ABCD(My, z_M, n1=n1)

        return qx, qy

    def profiles(self, z_start=None, z_end=None, clipping=None):
        elems = self.elems[:]
        ns = self.ns[:]

        if z_start is not None:
            elems.insert(0, Identity(z_start))
            ns.insert(0, ns[0])
        if z_end is not None:
            elems.append(Identity(z_end))
            ns.append(ns[-1])

        zs, Mxs, Mys, n_pairs = path_info(elems, ns)
        qx, qy = self._leftmost_qs(zs, Mxs, Mys, n_pairs)

        all_zs, x_profs, y_profs, x_RoCs, y_RoCs = [], [], [], [], []
        for Mx, My, (z_M, z_next_M), (n1, n2) in zip(Mxs, Mys, pairwise(zs), n_pairs):
            # FIXME: Clear up this wavlen/n business
            qx = qx and qx.apply_ABCD(Mx, z_M, n2=n2)
            qy = qy and qy.apply_ABCD(My, z_M, n2=n2)
            z = unitful_linspace(z_M, z_next_M, 1000)
            all_zs.append(z)
            x_profs.append(qx.profile(z, clipping))
            y_profs.append(qy.profile(z, clipping))
            x_RoCs.append(qx.roc(z))
            y_RoCs.append(qy.roc(z))
        return all_zs, x_profs, y_profs, x_RoCs, y_RoCs

    def plot(self, z_start=None, z_end=None, clipping=None, ax=None, mode='xy', **kwds):
        if mode not in ('x', 'y', 'xy'):
            raise ValueError("mode must be one of ('x', 'y', 'xy')")
        zs, x_profs, y_profs, _, _ = self.profiles(z_start, z_end, clipping)

        #if ax is None:
        #    from matplotlib.pyplot import subplots
        #    _, ax = subplots()
        import matplotlib.pyplot as plt
        zs_flat = chain_qty(*zs)
        x_profs_flat = chain_qty(*x_profs)
        y_profs_flat = chain_qty(*y_profs)

        plt_kwds = {'label': self.name}
        plt_kwds.update(kwds)
        if 'x' in mode:
            plt.plot(zs_flat, x_profs_flat, **plt_kwds)
        if 'y' in mode:
            plt.plot(zs_flat, y_profs_flat, **plt_kwds)

        ax = plt.gca()
        if not hasattr(ax, '_vlines'):
            ax._vlines = {}

        ymin, ymax = ax.get_ylim()
        for z in zs:
            zm0 = z.m[0]
            if zm0 in ax._vlines:
                ax._vlines[zm0].remove()

            vls = ax.vlines([zm0], 0, ymax,
                            linestyle='dashed', linewidth=2, color=(.5, .5, .5),
                            antialiased=True)
            ax._vlines[zm0] = vls
        xmin, xmax = ax.get_xlim()

        if hasattr(ax, '_hline'):
            ax._hline.remove()
        ax._hline = ax.hlines([0], xmin, xmax, linestyle='dotted', linewidth=2,
                              color='k', antialiased=True)

        ax.set_xlabel(f'Z Position [{x_profs_flat.units:~}]')
        ax.set_ylabel(f'Spot size [{y_profs_flat.units:~}]')

    def q_at(self, z):
        """The BeamParam that for the region that z lies within"""
        z = ensure_units(z, 'mm')
        zs, Mxs, Mys, n_pairs = path_info(self.elems, self.ns)
        qx, qy = self._leftmost_qs(zs, Mxs, Mys, n_pairs)

        for z_M, Mx, My, (n1, n2) in zip(zs, Mxs, Mys, n_pairs):
            if z < z_M:
                return (qx, qy)
            qx = qx and qx.apply_ABCD(Mx, z_M, n2=n2)
            qy = qy and qy.apply_ABCD(My, z_M, n2=n2)
        return (qx, qy)


def chain_qty(*quantities):
    qty0 = quantities[0]
    size = sum(len(qty) for qty in quantities)
    arr = np.empty((size,), 'float64')

    i = 0
    for qty in quantities:
        arr[i:i+len(qty)] = qty.magnitude
        i += len(qty)

    return Q_(arr, qty0.units)
