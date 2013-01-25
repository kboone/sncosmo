# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Utilities with functionality not specific to this package."""

import numpy as np

__all__ = ['GridData']

IRSA_BASE_URL = \
    'http://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={:.5f}+{:.5f}'

class GridData(object):
    """Interpolate over uniform 2-D grid.

    Similar to `scipy.interpolate.iterp2d` but with methods for returning
    native sampling.
    
    Parameters
    ----------
    x0 : numpy.ndarray
    x1 : numpy.ndarray
    y : numpy.ndarray
    """
    
    def __init__(self, x0, x1, y):
        self._x0 = np.asarray(x0)
        self._x1 = np.asarray(x1)
        self._y = np.asarray(y)
        self._yfuncs = []
        for i in range(len(self._x0)):
            self._yfuncs.append(lambda x: np.interp(x, self._x1, y[i,:]))

    def x0(self, copy=False):
        """Native x0 values."""
        if copy: return self._x0.copy()
        else: return self._x0

    def x1(self, copy=False):
        """Native x1 values."""
        if copy: return self._x1.copy()
        else: return self._x1

    def y(self, x0, x1=None, extend=True):
        """Return y values at requested x0 and x1 values.

        Parameters
        ----------
        x0 : float
        x1 : numpy.ndarray, optional
            Default value is None, which is interpreted as native x1 values.
        extend : bool, optional
            The function raises ValueError if x0 is outside of native grid,
            unless extend is True, in which case it returns values at nearest
            native x0 value.

        Returns
        -------
        yvals : numpy.ndarray
            1-D array of interpolated y values at requested x0 value and
            x1 values.
        """

        # Bounds check first
        if (x0 < self._x0[0] or x0 > self._x0[-1]) and not extend:
            raise ValueError("Requested x0 {:.2f} out of range ({:.2f}, "
                             "{:.2f})".format(x0, self._x0[0], self._x0[-1]))

        # Use default x1 if none are specified
        if x1 is None: x1 = self._x1

        # Check if requested x0 is out of bounds or exactly in the list
        if (self._x0 == x0).any():
            idx = np.flatnonzero(self._x0 == x0)[0]
            return self._yfuncs[idx](x1)
        elif x0 < self._x0[0]:
            return self._yfuncs[0](x1)
        elif x0 > self._x0[-1]:
            return self._yfuncs[-1](x1)
            
        # If we got this far, we need to interpolate between x0 values
        i = np.searchsorted(self._x0, x0)
        y0 = self._yfuncs[i - 1](x1)
        y1 = self._yfuncs[i](x1)
        dx0 = ((x0 - self._x0[i - 1]) /
               (self._x0[i] - self._x0[i - 1]))
        dy = y1 - y0
        return y0 + dx0 * dy


def mwebv(ra, dec, source='irsa'):
    """Return Milky Way E(B-V) at given coordinates.

    Parameters
    ----------
    ra : float
    dec : float
    source : {'irsa'}, optional
        Default is 'irsa', which means to make a web query of the IRSA 
        Schlegel dust map calculator. No other sources are currently
        supported.

    Returns
    -------
    mwebv : float
        Milky Way E(B-V) at given coordinates.
    """

    import urllib
    from xml.dom.minidom import parse

    # Check coordinates
    if ra < 0. or ra > 360. or dec < -90. or dec > 90.:
        raise ValueError('coordinates out of range')

    if source == 'irsa':
        try:
            u = urllib.urlopen(IRSA_BASE_URL.format(ra, dec))
            if not u:
                raise ValueError('URL query returned false')
        except:
            print 'E(B-V) web query failed'
            raise

        dom = parse(u)
        u.close()

        try:
            EBVstr = dom.getElementsByTagName('meanValue')[0].childNodes[0].data
            result = float(EBVstr.strip().split()[0])
        except:
            print "E(B-V) query failed. Do you have internet access?"
            raise

        return result
