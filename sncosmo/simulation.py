"""Tools for simulation of transients."""

import copy
import abc
import math
from collections import OrderedDict

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from numpy import random
from scipy.interpolate import InterpolatedUnivariateSpline as Spline1d
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from astropy.cosmology import WMAP9
from astropy import table
from astropy import units as u

from .utils import alias_map, integration_grid
from .models import Model


default_cosmo = WMAP9

# TODO: Figure out what should be in __all__
# __all__ = ['zdist', 'realize_lcs']

WHOLESKY_SQDEG = 4. * np.pi * (180. / np.pi) ** 2


def zdist(zmin, zmax, time=365.25, area=1.,
          ratefunc=lambda z: 1.e-4,
          cosmo=FlatLambdaCDM(H0=70.0, Om0=0.3)):
    """Generate a distribution of redshifts.

    Generates the correct redshift distribution and number of SNe, given
    the input volumetric SN rate, the cosmology, and the observed area and
    time.

    Parameters
    ----------
    zmin, zmax : float
        Minimum and maximum redshift.
    time : float, optional
        Time in days (default is 1 year).
    area : float, optional
        Area in square degrees (default is 1 square degree). ``time`` and
        ``area`` are only used to determine the total number of SNe to
        generate.
    ratefunc : callable
        A callable that accepts a single float (redshift) and returns the
        comoving volumetric rate at each redshift in units of yr^-1 Mpc^-3.
        The default is a function that returns ``1.e-4``.
    cosmo : `~astropy.cosmology.Cosmology`, optional
        Cosmology used to determine volume. The default is a FlatLambdaCDM
        cosmology with ``Om0=0.3``, ``H0=70.0``.

    Examples
    --------

    Loop over the generator:

    >>> for z in zdist(0.0, 0.25):
    ...     print(z)
    ...
    0.151285827576
    0.204078030595
    0.201009196731
    0.181635472172
    0.17896188781
    0.226561237264
    0.192747368762

    This tells us that in one observer-frame year, over 1 square
    degree, 7 SNe occured at redshifts below 0.35 (given the default
    volumetric SN rate of 10^-4 SNe yr^-1 Mpc^-3). The exact number is
    drawn from a Poisson distribution.

    Generate the full list of redshifts immediately:

    >>> zlist = list(zdist(0., 0.25))

    Define a custom volumetric rate:

    >>> def snrate(z):
    ...     return 0.5e-4 * (1. + z)
    ...
    >>> zlist = list(zdist(0., 0.25, ratefunc=snrate))

    """

    # Get comoving volume in each redshift shell.
    z_bins = 100  # Good enough for now.
    z_binedges = np.linspace(zmin, zmax, z_bins + 1)
    z_binctrs = 0.5 * (z_binedges[1:] + z_binedges[:-1])
    sphere_vols = cosmo.comoving_volume(z_binedges).value
    shell_vols = sphere_vols[1:] - sphere_vols[:-1]

    # SN / (observer year) in shell
    shell_snrate = np.array([shell_vols[i] *
                             ratefunc(z_binctrs[i]) / (1.+z_binctrs[i])
                             for i in range(z_bins)])

    # SN / (observer year) within z_binedges
    vol_snrate = np.zeros_like(z_binedges)
    vol_snrate[1:] = np.add.accumulate(shell_snrate)

    # Create a ppf (inverse cdf). We'll use this later to get
    # a random SN redshift from the distribution.
    snrate_cdf = vol_snrate / vol_snrate[-1]
    snrate_ppf = Spline1d(snrate_cdf, z_binedges, k=1)

    # Total numbe of SNe to simulate.
    nsim = vol_snrate[-1] * (time/365.25) * (area/WHOLESKY_SQDEG)

    for i in range(random.poisson(nsim)):
        yield float(snrate_ppf(random.random()))


OBSERVATIONS_ALIASES = OrderedDict([
    ('time', set(['time', 'date', 'jd', 'mjd', 'mjdobs', 'mjd_obs'])),
    ('band', set(['band', 'bandpass', 'filter', 'flt'])),
    ('zp', set(['zp', 'zpt', 'zeropoint', 'zero_point'])),
    ('zpsys', set(['zpsys', 'zpmagsys', 'magsys'])),
    ('gain', set(['gain'])),
    ('skynoise', set(['skynoise']))
])

OBSERVATIONS_REQUIRED_ALIASES = ('time', 'band', 'zp', 'zpsys', 'gain',
                                 'skynoise')


def realize_lcs(observations, model, params, thresh=None,
                trim_observations=False, scatter=True):
    """Realize data for a set of SNe given a set of observations.

    Parameters
    ----------
    observations : `~astropy.table.Table` or `~numpy.ndarray`
        Table of observations. Must contain the following column names:
        ``band``, ``time``, ``zp``, ``zpsys``, ``gain``, ``skynoise``.
    model : `~sncosmo.Model`
        The model to use in the simulation.
    params : list (or generator) of dict
        List of parameters to feed to the model for realizing each light curve.
    thresh : float, optional
        If given, light curves are skipped (not returned) if none of the data
        points have signal-to-noise greater than ``thresh``.
    trim_observations : bool, optional
        If True, only observations with times between
        ``model.mintime()`` and ``model.maxtime()`` are included in
        result table for each SN. Default is False.
    scatter : bool, optional
        If True, the ``flux`` value of the realized data is calculated by
        adding  a random number drawn from a Normal Distribution with a
        standard deviation equal to the ``fluxerror`` of the observation to
        the bandflux value of the observation calculated from model. Default
        is True.

    Returns
    -------
    sne : list of `~astropy.table.Table`
        Table of realized data for each item in ``params``.

    Notes
    -----
    ``skynoise`` is the image background contribution to the flux measurement
    error (in units corresponding to the specified zeropoint and zeropoint
    system). To get the error on a given measurement, ``skynoise`` is added
    in quadrature to the photon noise from the source.

    It is left up to the user to calculate ``skynoise`` as they see fit as the
    details depend on how photometry is done and possibly how the PSF is
    is modeled. As a simple example, assuming a Gaussian PSF, and perfect
    PSF photometry, ``skynoise`` would be ``4 * pi * sigma_PSF * sigma_pixel``
    where ``sigma_PSF`` is the standard deviation of the PSF in pixels and
    ``sigma_pixel`` is the background noise in a single pixel in counts.

    """

    RESULT_COLNAMES = ('time', 'band', 'flux', 'fluxerr', 'zp', 'zpsys')
    lcs = []

    # Copy model so we don't mess up the user's model.
    model = copy.copy(model)

    # get observations as a Table
    if not isinstance(observations, Table):
        if isinstance(observations, np.ndarray):
            observations = Table(observations)
        else:
            raise ValueError("observations not understood")

    # map column name aliases
    colname = alias_map(observations.colnames, OBSERVATIONS_ALIASES,
                        required=OBSERVATIONS_REQUIRED_ALIASES)

    # result dtype used when there are no observations
    band_dtype = observations[colname['band']].dtype
    zpsys_dtype = observations[colname['zpsys']].dtype
    result_dtype = ('f8', band_dtype, 'f8', 'f8', 'f8', zpsys_dtype)

    for p in params:
        model.set(**p)

        # Select times for output that fall within tmin amd tmax of the model
        if trim_observations:
            mask = ((observations[colname['time']] > model.mintime()) &
                    (observations[colname['time']] < model.maxtime()))
            snobs = observations[mask]
        else:
            snobs = observations

        # explicitly detect no observations and add an empty table
        if len(snobs) == 0:
            if thresh is None:
                lcs.append(Table(names=RESULT_COLNAMES,
                                 dtype=result_dtype, meta=p))
            continue

        flux = model.bandflux(snobs[colname['band']],
                              snobs[colname['time']],
                              zp=snobs[colname['zp']],
                              zpsys=snobs[colname['zpsys']])

        fluxerr = np.sqrt(snobs[colname['skynoise']]**2 +
                          np.abs(flux) / snobs[colname['gain']])

        # Scatter fluxes by the fluxerr
        # np.atleast_1d is necessary here because of an apparent bug in
        # np.random.normal: when the inputs are both length 1 arrays,
        # the output is a Python float!
        if scatter:
            flux = np.atleast_1d(np.random.normal(flux, fluxerr))

        # Check if any of the fluxes are significant
        if thresh is not None and not np.any(flux/fluxerr > thresh):
            continue

        data = [snobs[colname['time']], snobs[colname['band']], flux, fluxerr,
                snobs[colname['zp']], snobs[colname['zpsys']]]

        lcs.append(Table(data, names=RESULT_COLNAMES, meta=p))

    return lcs

def random_ra_dec(count=None):
    if count == None:
        use_count = 1
    else:
        use_count = count

    p, q = np.random.random((2, use_count))

    ra = 360. * p
    dec = np.arcsin(2. * (q - 0.5)) * 180 / np.pi

    if count == None:
        return ra[0], dec[0]
    else:
        return ra, dec


class Field(abc.ABC):
    """Abstract base class to represent a field on the sky."""
    @abc.abstractmethod
    def query(self, ra, dec):
        """Query whether an object at a specific ra and dec is in the field"""

    @abc.abstractmethod
    def sample(self, count=None):
        """Sample a random ra and dec from the field."""

    @property
    @abc.abstractmethod
    def solid_angle(self):
        """Return the solid angle of the field in square degrees"""


class FullSkyField(Field):
    def query(self, ra, dec):
        if np.isscalar(ra):
            return True
        else:
            return np.ones_like(ra, dtype=bool)

    def sample(self, count=None):
        return random_ra_dec(count)

    @property
    def solid_angle(self):
        return WHOLESKY_SQDEG


class BoxField(Field):
    def __init__(self, min_ra, max_ra, min_dec, max_dec):
        self.min_ra = min_ra
        self.max_ra = max_ra
        self.min_dec = min_dec
        self.max_dec = max_dec

    def query(self, ra, dec):
        return (
            (ra >= self.min_ra)
            & (ra < self.max_ra)
            & (dec >= self.min_dec)
            & (dec < self.max_dec)
        )

    def sample(self, count=None):
        raise NotImplementedError("Sample not implemented for BoxField! Aah!")

    @property
    def solid_angle(self):
        raise NotImplementedError("coverage not implemented for BoxField! Aah!")


class CircularField(Field):
    """TODO: circular field around a specific point on the sky"""


class SourceDistribution(abc.ABC):
    """Class to represent the distribution of sources across the sky"""
    @abc.abstractmethod
    def field_rate(self, field):
        """Calculate how many transients will be seen in a given field per year.

        Parameters
        ----------
        field : `~sncosmo.Field`
            The field that is being observed.
        """

    @abc.abstractmethod
    def simulate(self, count, start_time, end_time):
        """Simulate a source catalog

        TODO: Figure out what should actually go in the base class.
        """


class VolumetricSourceDistribution(SourceDistribution):
    """Model sources that are evenly distributed across the sky.

    Parameters
    ----------
    volumetric_rate : float or function
        The volumetric rate in counts/yr/Mpc**3. This can be either a float
        representing a constant rate as a function of redshift or a function
        that takes the redshift as a parameter and returns the volumetric rate
        at that redshift.
    """
    def __init__(self, volumetric_rate=None, min_redshift=0., max_redshift=3.,
                 cosmo=default_cosmo):
        if volumetric_rate is not None:
            self.volumetric_rate = volumetric_rate
        self.cosmo = cosmo

        self._update_redshift_distribution(min_redshift, max_redshift)

    def _update_redshift_distribution(self, min_redshift, max_redshift,
                                      redshift_sampling=0.001):
        """Set up the distribution to operate over a given redshift range.

        This also creates an inverse CDF sampler to draw redshifts from.
        """
        self.min_redshift = min_redshift
        self.max_redshift = max_redshift

        if self.min_redshift == self.max_redshift:
            # Simulating targets at a single redshift. The rates are meaningless in this
            # case and we just always sample at that redshift.
            self.redshift_cdf = lambda x: self.min_redshift
            return

        # Sample the rates to build a CDF for inverse transform sampling.

        # Figure out the redshift range to use for sampling the PDF. We use the number
        # of bins that gives a sampling closest to, but less than, redshift_sampling.
        z_range = self.max_redshift - self.min_redshift
        sample_count = int(math.ceil(z_range / redshift_sampling)) + 1
        sample_z = np.linspace(self.min_redshift, self.max_redshift, sample_count)

        # Figure out the number of sources that we would expect to see over the entire
        # sky as a function of redshift.
        all_sky_rates = self.all_sky_rate(sample_z)

        # Turn this into a CDF of rates.
        cum_rates = np.cumsum(all_sky_rates) / np.sum(all_sky_rates)

        self.redshift_cdf = interpolate.interp1d(
            cum_rates,
            sample_z
        )

    def all_sky_rate(self, redshift):
        """Calculate the all sky rate of this source at a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift to estimate the rate at.

        Returns
        -------
        all_sky_rate : float
            The rate of this source at the given redshift in counts/year/unit redshift
        """
        if callable(self.volumetric_rate):
            rate = self.volumetric_rate(redshift)
        else:
            rate = self.volumetric_rate

        rate = (
            4. * np.pi
            * rate
            * self.cosmo.differential_comoving_volume(redshift).value 
        )

        return rate

    def field_rate(self, field):
        """Calculate how many transients will be seen in a given field per year.

        Parameters
        ----------
        field : `~sncosmo.Field`
            The field that is being observed.
        """
        solid_angle = field.solid_angle
        all_sky_rate = integrate.quad(self.all_sky_rate, self.min_redshift,
                                      self.max_redshift)[0]

        return solid_angle / WHOLESKY_SQDEG * all_sky_rate

    def simulate(self, field, start_time, end_time, count, flat_redshift=False):
        """Simulate a catalog"""
        ref_count = self.field_rate(field) * (end_time - start_time) * 365.25

        # TODO: Weighting by redshift or things like that
        # if flat_redshift:
            # Sample from a flat redshift distribution
            # redshifts = np.random.uniform(self.min_redshift, self.max_redshift,
                                          # count)
            # weights = self.all_sky_rate(redshifts) / count
        # else:
            # Sample from the True redshift distribution

        redshifts = self.redshift_cdf(np.random.random(size=count))
        weights = np.ones(len(redshifts))

        ras, decs = field.sample(count)
        t0 = np.random.uniform(start_time, end_time, count)

        result = table.Table({
            'z': redshifts,
            't0': t0,
            'ra': ras,
            'dec': decs,
            'weight': weights,
        })

        model, parameters = self.simulate_parameters(result)
        result['model'] = model
        result['parameters'] = _generate_parameter_dicts(
            z=result['z'], t0=result['t0'], **parameters
        )

        return result

    @abc.abstractmethod
    def simulate_parameters(self, locations):
        """Simulate model parameters given the locations of objects on the sky.

        This should be implemented in subclasses.

        Parameters
        ----------
        location : `~astropy.table.Table`
            A table containing the locations (ra, dec, z, t0) of all of the objects to
            simulate.

        Returns
        -------
        model : Model or list of models.
            The Model to use for each entry. This can be either a single Model object or
            a list of model objects.

        parameters : dict
            A dictionary with all of the parameters to set. The keys should be the names
            of the parameters, and the values should be a list of the different
            parameter values for each object that is being simulated.
        """


def _model_amplitude_zp(model, band, magsys, cosmo, ref_z=0.01):
    """Determine the amplitude zeropoint for an absolute magnitude of zero.

    Some models, like SALT2, have an arbitrary zeropoint for their templates.
    """
    # Choose an arbitrary redshift to evaluate things at.
    model.set(z=ref_z)
    model.set_source_peakabsmag(0., band, magsys, cosmo=cosmo)
    amplitude_zp = -2.5*np.log10(model.parameters[2]) - cosmo.distmod(ref_z).value

    return amplitude_zp


class SALT2Distribution(VolumetricSourceDistribution):
    """Model the distribution of Type Ia supernovae across the sky using the
    SALT2 model.
    """
    ref_absmag = -19.1
    ref_band = 'bessellb'
    ref_magsys = 'ab'

    alpha = 0.13
    beta = 3.1
    sigma_int = 0.1

    def volumetric_rate(self, redshift):
        return 2.6e-5 * (1 + redshift)

    def simulate_parameters(self, locations):
        model = Model(source='salt2-extended')
        count = len(locations)

        x1 = np.random.normal(0, 1, count)
        c = np.random.exponential(0.1, count)

        # Figure out the value of x0 to use. We determine the required offset so that a
        # supernova with x1=0 and c=0 has an absolute magnitude of ref_mag. This is
        # cosmology depdendent, and will be affected by the value H0.
        mag_x0_zp = _model_amplitude_zp(model, self.ref_band, self.ref_magsys,
                                        self.cosmo)
        mag_x0 = (
            self.ref_absmag
            + self.alpha * x1
            - self.beta * c
            + np.random.normal(0, self.sigma_int, count)
            + self.cosmo.distmod(locations['z']).value
            + mag_x0_zp
        )
        x0 = 10**(-0.4 * mag_x0)

        parameters = {'x0': x0, 'x1': x1, 'c': c}

        return model, parameters


class TemplateVolumetricDistribution(VolumetricSourceDistribution):
    """Model the volumetric distribution of a single basic template.

    We assume that the template only has three parameters: z, t0 and a parameter that
    represents the amplitude of the template. This is the case for any of the
    `~sncosmo.TimeSeriesSource` templates.
    """
    def __init__(self, source, volumetric_rate, ref_absmag, ref_absmag_dispersion, 
                 ref_band='bessellb', ref_magsys='ab', *args, **kwargs):
        super().__init__(volumetric_rate, *args, **kwargs)
        self.model = Model(source=source)
        self.ref_absmag = ref_absmag
        self.ref_absmag_dispersion = ref_absmag_dispersion
        self.ref_band = ref_band
        self.ref_magsys = ref_magsys

    def simulate_parameters(self, locations):
        count = len(locations)

        amplitude_zp = _model_amplitude_zp(self.model, self.ref_band, self.ref_magsys,
                                           self.cosmo)
        mag_amplitude = (
            self.ref_absmag
            + np.random.normal(0, self.ref_absmag_dispersion, count)
            + amplitude_zp
        )
        amplitude = 10**(-0.4 * mag_amplitude)
        amplitude_parameter_name = self.model.param_names[2]

        parameters = {amplitude_parameter_name: amplitude}

        return self.model, parameters


def _generate_parameter_dicts(**kwargs):
    """Generate individual parameter dicts for a large number of sources

    Parameters
    ----------
    kwargs
        For each key, a list of the values for that key.
    """
    parameter_dicts = []
    keys = kwargs.keys()
    for row_values in zip(*kwargs.values()):
        parameter_dicts.append(dict(zip(keys, row_values)))

    return parameter_dicts


def cc_volumetric_rate(z):
    z = np.atleast_1d(z)

    result = np.zeros(shape=np.shape(z))

    low_z_mask = z < 0.8

    result[low_z_mask] = (5.0e-5*(1+z[low_z_mask])**4.5)
    result[~low_z_mask] = 5.44e-4

    if np.ndim(z) == 0:
        return result[0]
    else:
        return result
