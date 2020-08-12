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

def random_ra_dec(count=None, min_ra=0., max_ra=360., min_dec=-90., max_dec=90.):
    """Return a random RA and Dec with the given bounds

    By default, this uses the full sky.
    """
    if count == None:
        use_count = 1
    else:
        use_count = count

    p, q = np.random.random((2, use_count))

    sin_min_dec = np.sin(min_dec * np.pi / 180.)
    sin_max_dec = np.sin(max_dec * np.pi / 180.)

    ra = min_ra + p * (max_ra - min_ra)
    dec = np.arcsin((sin_max_dec - sin_min_dec) * q + sin_min_dec) * 180 / np.pi

    if count == None:
        return ra[0], dec[0]
    else:
        return ra, dec

class Field(abc.ABC):
    """Abstract base class to represent a field on the sky.

    A field captures the area on the sky that should be simulated along with the time
    period that the simulation should cover.
    """
    @abc.abstractmethod
    def query(self, ra, dec, time):
        """Query whether an object at a specific ra and dec is in the field"""

    @abc.abstractmethod
    def sample(self, count=None):
        """Sample a random ra and dec and time from the field."""

    @property
    @abc.abstractmethod
    def solid_angle(self):
        """Return the solid angle of the field in square degrees"""

    @property
    @abc.abstractmethod
    def solid_angle_time(self):
        """Return the solid angle-time of the field in square degree years"""

    def __add__(self, other):
        return CombinedField(self, other)


class FullSkyField(Field):
    def __init__(self, min_time, max_time):
        self.min_time = min_time
        self.max_time = max_time

    def query(self, ra, dec, time):
        return (time >= self.min_time) & (time < self.max_time)

    def sample(self, count=None):
        ra, dec = random_ra_dec(count)
        time = np.random.uniform(self.min_time, self.max_time, count)
        return ra, dec, time

    @property
    def solid_angle(self):
        return WHOLESKY_SQDEG

    @property
    def solid_angle_time(self):
        return self.solid_angle * (self.max_time - self.min_time) / 365.25


class BoxField(Field):
    def __init__(self, min_time, max_time, min_ra, max_ra, min_dec, max_dec):
        self.min_ra = min_ra
        self.max_ra = max_ra
        self.min_dec = min_dec
        self.max_dec = max_dec
        self.min_time = min_time
        self.max_time = max_time

    def query(self, ra, dec, time):
        return (
            (ra >= self.min_ra)
            & (ra < self.max_ra)
            & (dec >= self.min_dec)
            & (dec < self.max_dec)
            & (time >= self.min_time)
            & (time < self.max_time)
        )

    def sample(self, count=None):
        ra, dec = random_ra_dec(count, self.min_ra, self.max_ra, self.min_dec,
                                self.max_dec)
        time = np.random.uniform(self.min_time, self.max_time, count)
        return ra, dec, time

    @property
    def solid_angle(self):
        return (
            (self.max_ra - self.min_ra)
            * (np.sin(self.max_dec * np.pi / 180.)
               - np.sin(self.min_dec * np.pi / 180.))
            * (180. / np.pi)
        )

    @property
    def solid_angle_time(self):
        return self.solid_angle * (self.max_time - self.min_time) / 365.25


class CircularField(Field):
    """TODO: circular field around a specific point on the sky"""


class CombinedField(Field):
    """Class to represent the combination of multiple fields

    Parameters
    *fields : `~sncosmo.Field` objects
        The fields to combine
    """
    def __init__(self, *fields):
        use_fields = []
        for field in fields:
            # Make sure that we were actually given Field objects
            if not isinstance(field, Field):
                raise ValueError("can only combine fields.")

            # Unpack any other Field objects.
            if isinstance(field, CombinedField):
                use_fields.extend(field._fields)
            else:
                use_fields.append(field)

        self._fields = use_fields

    def query(self, ra, dec, time):
        # Check each of the subfields individually.
        result = False
        for field in self._fields:
            result &= field.query(ra, dec, time)
        return result

    def sample(self, count=None):
        """Sample a random ra and dec and time from one of the subfields.

        Each subfield is weighted based off of the solid_angle_time. This is correct for
        transients, but if non-transient sources (e.g. variable stars) are ever
        implemented, they should weight by solid_angle instead.
        """
        if count is None:
            single = True
            count = 1
        else:
            single = False

        # Figure out the probability of each source type
        weights = np.array([i.solid_angle_time for i in self._fields])
        norm_weights = weights / np.sum(weights)

        # Randomly choose how many of each field to use.
        fields = np.random.choice(np.arange(len(self._fields)), count, p=norm_weights)

        # Simulate the desired number of each field.
        ra = []
        dec = []
        time = []
        for field_id, field in enumerate(self._fields):
            field_count = np.sum(fields == field_id)
            if field_count > 0:
                field_ra, field_dec, field_time = field.sample(field_count)
                ra.append(field_ra)
                dec.append(field_dec)
                time.append(field_time)

        ra = np.hstack(ra)
        dec = np.hstack(dec)
        time = np.hstack(time)

        # Randomly shuffle the positions.
        order = np.arange(count)
        np.random.shuffle(order)
        ra = ra[order]
        dec = dec[order]
        time = time[order]

        if single:
            return ra[0], dec[0], time[0]
        else:
            return ra, dec, time

    @property
    def solid_angle(self):
        solid_angle = 0
        for field in self._fields:
            solid_angle += field.solid_angle
        return solid_angle

    @property
    def solid_angle_time(self):
        solid_angle_time = 0
        for field in self._fields:
            solid_angle_time += field.solid_angle_time
        return solid_angle_time


class SourceDistribution(abc.ABC):
    """Class to represent the distribution of sources across the sky"""
    @abc.abstractmethod
    def field_count(self, field):
        """Calculate how many transients will be seen in a given field.

        Parameters
        ----------
        field : `~sncosmo.Field`
            The field that is being observed.
        """

    @abc.abstractmethod
    def simulate(self, field, count, reference_count=None, **kwargs):
        """Simulate a source catalog

        TODO: Figure out what should actually go in the base class.
        """

    def __add__(self, other):
        return CombinedSourceDistribution(self, other)


class CombinedSourceDistribution(SourceDistribution):
    """Class to represent the combination of multiple different kinds of sources

    Parameters
    ----------
    *source_distributions : `~sncosmo.SourceDistribution` objects
        The source distributions to combine.
    """
    def __init__(self, *source_distributions):
        use_distributions = []
        for source_distribution in source_distributions:
            # Make sure that we were actually given SourceDistribution objects.
            if not isinstance(source_distribution, SourceDistribution):
                raise ValueError("can only combine source distributions.")

            # Unpack any other CombinedSourceDistribution objects.
            if isinstance(source_distribution, CombinedSourceDistribution):
                use_distributions.extend(source_distribution._source_distributions)
            else:
                use_distributions.append(source_distribution)

        self._source_distributions = use_distributions

    def field_count(self, field):
        """Calculate how many transients will be seen in a given field.

        This counts all of the different kinds of sources.

        Parameters
        ----------
        field : `~sncosmo.Field`
            The field that is being observed.
        """
        total_count = 0
        for source_distribution in self._source_distributions:
            total_count += source_distribution.field_count(field)

        return total_count

    def simulate(self, field, count, reference_count=None, **kwargs):
        """Simulate a source catalog

        This samples from all of the different sub-distributions in proportion to their
        rates.
        """
        # Figure out the probability of each source type
        counts = np.array([i.field_count(field) for i in self._source_distributions])
        norm_counts = counts / np.sum(counts)

        if reference_count is None:
            reference_count = np.sum(counts)

        # Randomly choose how many of each model to use.
        models = np.random.choice(np.arange(len(self._source_distributions)), count,
                                  p=norm_counts)

        # Simulate the desired number of each model.
        full_catalog = []
        for source_id, source_distribution in enumerate(self._source_distributions):
            source_count = np.sum(models == source_id)
            source_reference_count = source_count / count * reference_count
            source_catalog = source_distribution.simulate(
                field,
                source_count,
                source_reference_count,
                **kwargs
            )
            full_catalog.append(source_catalog)

        full_catalog = table.vstack(full_catalog)

        # Randomly shuffle the catalog.
        order = np.arange(count)
        np.random.shuffle(order)
        full_catalog = full_catalog[order]

        return full_catalog


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

    def field_count(self, field):
        """Calculate how many transients will be seen in a given field.

        Parameters
        ----------
        field : `~sncosmo.Field`
            The field that is being observed.
        """
        solid_angle_time = field.solid_angle_time
        all_sky_count = integrate.quad(self.all_sky_rate, self.min_redshift,
                                       self.max_redshift)[0]

        return solid_angle_time / WHOLESKY_SQDEG * all_sky_count

    def simulate(self, field, count, reference_count=None, flat_redshift=False,
                 **kwargs):
        """Simulate a catalog"""
        if reference_count is None:
            reference_count = self.field_count(field)

        if flat_redshift:
            # Sample from a flat redshift distribution
            redshifts = np.random.uniform(self.min_redshift, self.max_redshift, count)
            raw_weights = self.all_sky_rate(redshifts)
        else:
            # Sample from the True redshift distribution
            redshifts = self.redshift_cdf(np.random.random(size=count))
            raw_weights = np.ones(len(redshifts))

        weights = raw_weights / np.sum(raw_weights) * reference_count

        ras, decs, times = field.sample(count)

        result = table.Table({
            'z': redshifts,
            't0': times,
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

    Additionally, we assume that the absolute magnitude of the template should be drawn
    from a Gaussian distribution.
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
            + self.cosmo.distmod(locations['z']).value
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
