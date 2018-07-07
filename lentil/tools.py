# -*- coding: utf-8 -*-
# Copyright 2013-2018 Nate Bogdanowicz
import numpy as np
from functools import reduce
from numpy import sqrt, complex, sign, pi
from scipy.special import erfinv
from .elements import OpticalElement, Identity
from .util import Q_, pairwise, unitful_linspace
from .util import u, Q_, pairwise, unitful_linspace, argrelmin


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
        self.lambda0 = Q_(wavlen).to('nm') * n
        self.n = n
        z0 = Q_(z0).to('mm')
        zR = Q_(zR).to('mm')
        self.q0 = -z0 + 1j * zR  # q(z=0)

    @classmethod
    def from_q(cls, q, wavlen, z='0mm', n=1):
        z = Q_(z).to('mm')
        zR = _get_imag(q)
        z0 = z - _get_real(q)
        return BeamParam(wavlen, zR, z0, n)

    @classmethod
    def from_waist(cls, wavlen, w0, z0='0mm', n=1):
        wavlen = Q_(wavlen).to('nm')
        w0 = Q_(w0).to('mm')
        z0 = Q_(z0).to('mm')
        zR = pi * w0**2 / wavlen
        return BeamParam(wavlen, zR, z0, n)

    @classmethod
    def from_widths(cls, wavlen, z1, w1, z2, w2, n=1, focus_location='left'):
        wavlen = Q_(wavlen).to('nm')
        z1 = Q_(z1).to('mm')
        z2 = Q_(z2).to('mm')
        w1 = Q_(w1).to('mm')
        w2 = Q_(w2).to('mm')

        if focus_location == 'left':
            zR_sign = +1
            z0_sign = -1
        elif focus_location == 'right':
            zR_sign = +1
            z0_sign = +1
        elif focus_location == 'between':
            zR_sign = -1
            z0_sign = -1 if z1 < z2 else +1
        else:
            raise ValueError
        # NOTE: wavlen should be wavlen in medium
        # zR expression derived using Wolfram Cloud
        zR_num = (w1**2 + w2**2) + zR_sign * 2*sqrt(w1**2*w2**2 - ((wavlen/pi*(z1-z2))**2))
        zR_den = pi/wavlen*((w1**2-w2**2)/(z1-z2))**2 + 4*wavlen/pi
        zR = zR_num/zR_den
        z0 = z1 + z0_sign * sqrt(zR * (pi/wavlen*w1**2 - zR))
        return cls(wavlen, zR, z0)

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
        lambda1 = self.lambda0 / n1
        A, B, C, D = M.elems()
        q2 = self.q(z)
        q1 = n1 * (D*q2/n2 - B) / (-C*q2/n2 + A)
        return BeamParam.from_q(q1, z=z, wavlen=lambda1, n=n1)

    def apply_optical_elements(self, elems, tangential=True, wavlen=None, n=1):
        """Make a new BeamParam by passing this BeamParam through a list of optical elements"""
        elems, ns = extract_elems_ns(elems, n)
        zs, Ms_tan, Ms_sag, n_pairs = path_info(elems, ns)
        Ms = Ms_tan if tangential else Ms_sag
        q = self
        for M, z_M, (n1, n2) in zip(Ms, zs, n_pairs):
            q = q.apply_ABCD(M, z_M, n2=n2)

        return q




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
            q = q.apply_ABCD(M, z_M)

            # Calculate beam info within region
            z = unitful_linspace(el.z, next_el.z, 1000)
            zs.append(z)
            profiles.append(q.profile(z, el.n, clipping))
            RoCs.append(q.roc(z))

    return zs, profiles, RoCs


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
        zs, Ms_tan, Ms_sag, n_pairs = path_info(self.elems, self.ns, z_start, z_end)
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

    def find_cavity_modes(self, wavlen, n=1, ring=True):
        """Find the eigenmodes of an optical cavity defined by this BeamPath

        The first and last elements of this BeamPath should be mirrors.

        Parameters
        ----------
        wavlen : Quantity([Length])
          The wavelength of the beam in this medium
        ring : bool
            If True, solve as a ring cavity. Otherwise, assumes the beam travels back through the
            optical elements, in a standing-wave configuration (like a Fabry-Perot cavity).

        Returns
        -------
        qt, qs : BeamParams
            beam parameter for the tangential and sagittal modes, respectively.
        """
        # FIXME: Verify we're doing the right thing regarding wavelengths and n's
        wavlen = Q_(wavlen).to('nm')
        zs, Ms_tan, Ms_sag, n_pairs = path_info(self.elems, self.ns, add_spaces=True)

        if not ring:
            Ms_tan = Ms_tan + Ms_tan[-2:0:-1]  # ABCDE -> ABCDEDCB
            Ms_sag = Ms_sag + Ms_sag[-2:0:-1]  # ABCDE -> ABCDEDCB

        qt_r = _find_cavity_mode(combine(Ms_tan))
        qs_r = _find_cavity_mode(combine(Ms_sag))

        qt = BeamParam.from_q(1/qt_r, wavlen=wavlen, n=n, z=zs[0])
        qs = BeamParam.from_q(1/qs_r, wavlen=wavlen, n=n, z=zs[0])
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

        t_kwds = dict(color='b', linewidth=3)
        t_kwds.update(kwds)
        s_kwds = dict(color='r', linewidth=3)
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


