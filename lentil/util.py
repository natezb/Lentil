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


def argrelmin(data):
    """ Finds indices of relative minima. Doesn't count first and last points
    as minima """
    args = []
    curmin = data[0]  # This keeps the first point from being included
    curminarg = None
    for i, num in enumerate(data):
        if num < curmin:
            curmin = num
            curminarg = i
        elif num > curmin and curminarg is not None:
            # Once curve starts going up, we know we've found a minimum
            args.append(curminarg)
            curminarg = None
    return args
