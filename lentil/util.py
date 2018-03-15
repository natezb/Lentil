# -*- coding: utf-8 -*-
# Copyright 2018 Nate Bogdanowicz
import itertools

from numpy import linspace
from pint import _DEFAULT_REGISTRY
u = _DEFAULT_REGISTRY
Q_ = u.Quantity


try:
    izip = itertools.izip
except AttributeError:
    izip = zip  # Python 3


def pairwise(seq):
    """Produce a sequence of adjacent pairs from the input sequence"""
    a, b = itertools.tee(seq)
    next(b, None)
    return izip(a, b)


def unitful_linspace(start, stop, *args, **kwds):
    start, stop = Q_(start), Q_(stop)
    units = start.units
    raw_pts = linspace(start.m, stop.m_as(units), *args, **kwds)
    return Q_(raw_pts, units)
