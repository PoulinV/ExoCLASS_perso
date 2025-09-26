u"""
.. module:: transfer
   :synopsis: Definition of the transfer-class and the laod and dump function
.. moduleauthor:: Patrick Stoecker <stoecker@physik.rwth-aachen.de>

Contains the definition of the transfer class :class:`transfer <DarkAgees.transfer.transfer>`
to store the 3D-array of the discretized transfer functions :math:`T_{klm}` and the
1D-array with the values of :math:`z_\\mathrm{dep}`, :math:`\\log_{10} E`, and :math:`z_\\mathrm{inj}`

Also contains methods to store (dump) an instance of this class in a file and
to load them from this.
"""

from __future__ import absolute_import, division, print_function
from builtins import object
from copy import deepcopy as _dcp
from scipy.integrate import trapz
import numpy as np
import dill

import os
import sys

class spectral_distortions(object):
    u"""
    Container of the discretized transfer functions :math:`T_{klm}` and the
    arrays with the values at which they are defined.

    Reads the transfer functions and pojts at which they are defined and stores them
    as numpy-arrays.
    """

    def __init__(self, infile):
        u"""
        Parameters
        ----------
        infile : :obj:`str`
        Path to the table z_deposited, log10E, z_injected, transfer_elec and transfer_phot
        in increasing order.
        """

        #print 'Initializing the transfer functions'
        data = np.genfromtxt(infile, unpack=True, usecols=(0,1,2,3,4), dtype=np.float64 )
        self.z_injected = np.unique(data[0]).astype(np.float64)
        self.frequency = np.unique(data[1]).astype(np.float64)
        self.E_injected = np.unique(data[2]).astype(np.float64)
        l1 = len(self.z_injected)
        l2 = len(self.frequency)
        l3 = len(self.E_injected)
        self.spectral_distortions_phot = data[4].reshape(l1,l2,l3).astype(np.float64)
        self.spectral_distortions_elec = data[3].reshape(l1,l2,l3).astype(np.float64)

    def __add__(self,other):
        returned_instance = _dcp(self)
        returned_instance.spectral_distortions_elec += other.spectral_distortions_elec
        returned_instance.spectral_distortions_phot += other.spectral_distortions_phot
        return returned_instance

    def __sub__(self,other):
        return self + (-other)

    def __neg__(self):
        negself = _dcp(self)
        negself.spectral_distortions_phot = -self.spectral_distortions_phot
        negself.spectral_distortions_elec = -self.spectral_distortions_elec
        return negself

    def __eq__(self,other):
        same = (self.spectral_distortions_elec.shape == other.spectral_distortions_elec.shape)
        if same:
            same = same & np.all(self.spectral_distortions_elec == other.spectral_distortions_elec)
            same = same & np.all(self.spectral_distortions_phot == other.spectral_distortions_phot)
        return same

def spectral_distortions_dump(spectral_distortions_instance, outfile):
	u"""Stores a initialized instance of the :class:`spectral_distortions <DarkAges.spectral_distortions.spectral_distortions>`
	-class in file using the dump method of :class:`dill`.

	Parameters
	----------
	spectral_distortions_instance : :obj:`class`
		Initialized instance of the :class:`spectral_distortions <DarkAges.spectral_distortions.spectral_distortions>`-class
	outfile : :obj:`str`
		Filename (absolute or relative) under which the spectral_distortions instance should be stored
	"""

	if not isinstance(spectral_distortions_instance, spectral_distortions):
		from .__init__ import DarkAgesError
		raise DarkAgesError('You did not include a proper instance of the class "spectral_distortions"')
	with open(outfile, 'wb') as f_dump:
		dill.dump(spectral_distortions_instance, f_dump)
	return

def spectral_distortions_load(infile):
	u"""Reloads an instance of the :class:`spectral_distortions <DarkAges.spectral_distortions.spectral_distortions>`
	-class dumped with :meth:`spectral_distortions_dump <DarkAges.spectral_distortions.spectral_distortions_dump>`

	Parameters
	----------
	infile : :obj:`str`
		Filename (absolute or relative) under which the spectral_distortions instance is stored

	Returns
	-------
	:obj:`class`
		Restored instance of the :class:`spectral_distortions <DarkAges.spectral_distortions.spectral_distortions>`-class
	"""

	loaded_spectral_distortions = dill.load(open(infile, 'rb'))
	#if not isinstance(loaded_spectral_distortions, spectral_distortions):
	#	from .__init__ import DarkAgesError
	#	raise DarkAgesError('The file {0} does not provide a proper instance of the class "spectral_distortions"'.format(infile))
	#else:
	return loaded_spectral_distortions

def spectral_distortion_today(frequency,z_injected, E_injected,transfer_functions_E,spectral_distortions_phot,spectral_distortions_elec,spec_elec, spec_phot,hist,normalization, sigmav=3e-26,t_dec=np.inf,n_cdm=0,**DarkOptions):
    # u"""Returns the effective efficiency factor :math:`f_c (z)`
    # for the deposition channel :math:`c`.
    #
    # This method calculates the effective efficiency factor in dependence
    # of the redshift
    #
    # In doing so the integral given in eq. (2.14) of `1801.01871 <https://arxiv.org/abs/1801.01871>`_ is performed.
    #
    # Parameters
    # ----------
    # normalization : :obj:`array-like`
    # 	Array (:code:`shape = (k)`) containing the proper normalization of the injected spectra
    # 	of photons and electrons at each timestep / at each redshift of deposition
    # spec_phot : :obj:`array-like`
    # 	Array (:code:`shape = (l',m)`) containing the double differential spectrum
    # 	:math:`\\frac{\\mathrm{d}^2 N}{ \\mathrm{d}E \\mathrm{d}t }` of photons
    # spec_elec : :obj:`array-like`
    # 	Array (:code:`shape = (l',m)`) containing the double differential spectrum
    # 	:math:`\\frac{\\mathrm{d}^2 N}{ \\mathrm{d}E \\mathrm{d}t }` of electrons
    # 	and positrons.
    #
    #
    # Returns
    # -------
    # :obj:`array-like`
    # 	Array (:code:`shape = (k)`) of :math:`E0dNdE0dV0` at the frequency today given in :code:`frequency`
    # """
    E = 10**(E_injected)
    dlogz = np.diff(np.log(z_injected))
    # print(dz,z_injected)
    # dlogz = np.append(dz, dz[0])

    #PROBLEME WITH E: should feed the injected particle energy AND the energy of the transfer function table separately.
    how_to_integrate = DarkOptions.get('E_integration_scheme','energy')
    if how_to_integrate not in ['logE','energy']:
        from .__init__ import DarkAgesError
        raise DarkAgesError('The energy integration-scheme >> {0} << is not known'.format(how_to_integrate))
    if len(E) == 1: how_to_integrate = 'energy' # Handling of a dirac-spectrum is inside the integration part w.r.t energy
    # norm = ( conversion(z_dep,alpha=alpha) )*( normalization )

    if (len(E_injected) == len(transfer_functions_E)):
    	if np.any(abs(log10E - transfer_functions_log10E) <= 1e-9*log10E):
    		need_to_interpolate = False
    	else:
    		from .common import evaluate_spectral_distortion_transfer
    		need_to_interpolate = True
    else:
    	from .common import evaluate_spectral_distortion_transfer
    	need_to_interpolate = True
    # need_to_interpolate = False
    energy_integral = np.zeros( shape=(len(frequency),len(z_injected)), dtype=np.float64)
    Enj = transfer_functions_E
    for i in range(len(frequency)): ##loop over frequency
        # print(frequency[i])
        if how_to_integrate == 'logE':
            for k in range(len(z_injected)): ##loop over z_inj
                if not need_to_interpolate:
                    # int_phot = spectral_distortions_phot[k,i,:]*spec_phot[:,k]*(E[:]**2)/np.log10(np.e)
                    # int_elec = spectral_distortions_elec[k,i,:]*spec_elec[:,k]*(E[:]**2)/np.log10(np.e)
                    int_phot = spectral_distortions_phot[k,i,:]*spec_phot[:,k]*(E[:]**2)/np.log10(np.e)
                    int_elec = spectral_distortions_elec[k,i,:]*spec_elec[:,k]*(E[:]**2)/np.log10(np.e)
                else:
                    int_phot = evaluate_spectral_distortion_transfer(Enj,spectral_distortions_phot[k,i,:],E)*spec_phot[:,k]*(E[:]**2)/np.log10(np.e)
                    int_elec = evaluate_spectral_distortion_transfer(Enj,spectral_distortions_elec[k,i,:],E)*spec_elec[:,k]*(E[:]**2)/np.log10(np.e)
                energy_integral[i][k] = trapz( int_phot + int_elec, log10E )
        elif how_to_integrate == 'energy':
            for k in range(len(z_injected)):
                if not need_to_interpolate:
                    # print(spectral_distortions_phot[k,i,:],spec_elec[:,k],(E[:]**1))
                    int_phot = spectral_distortions_phot[k,i,:]*spec_phot[:,k]
                    int_elec = spectral_distortions_elec[k,i,:]*spec_elec[:,k]
                    # int_phot = spectral_distortions_phot[k,i,:]*spec_phot[:,k]*(E[:]**1)
                    # int_elec = spectral_distortions_elec[k,i,:]*spec_elec[:,k]*(E[:]**1)
                else:
                    int_phot = evaluate_spectral_distortion_transfer(Enj,spectral_distortions_phot[k,i,:],E)*spec_phot[:,k]/2
                    int_elec = evaluate_spectral_distortion_transfer(Enj,spectral_distortions_elec[k,i,:],E)*spec_elec[:,k]/2
                    # int_elec = evaluate_spectral_distortion_transfer(Enj,spectral_distortions_elec[k,i,:],E)*2
                    # int_phot = evaluate_spectral_distortion_transfer(Enj,spectral_distortions_phot[k,i,:],E)*spec_phot[:,k]*(E[:]**1)
                    # int_elec = evaluate_spectral_distortion_transfer(Enj,spectral_distortions_elec[k,i,:],E)*spec_elec[:,k]*(E[:]**1)
                if len(E) > 1:
                    energy_integral[i][k] = trapz( int_phot + int_elec, E )
                else:
                    energy_integral[i][k] = int_phot + int_elec
                # if(energy_integral[i][k]==0):
                    # print('nu %f, z %f, E %f, dNdE %f, jntegral %f'%(frequency[i],z_injected[k],E[:],spec_elec[:,k],energy_integral[i][k]))
    result = np.zeros_like( frequency, dtype=np.float64)
    result2 = np.zeros_like( frequency, dtype=np.float64)
    rate_per_volume_per_time = np.zeros_like( z_injected, dtype=np.float64)
    if hist=='decay':
        rate_per_volume_per_time = n_cdm * (z_injected[:]+1)**3 / t_dec
    if hist=='annihilation':
        # print(DarkOptions.get('n_cdm'),z_injected[:],DarkOptions.get('sigmav'))
        rate_per_volume_per_time = (n_cdm*(z_injected[:]+1)**3) **2 * sigmav
    for i in range(len(result)):
        from .common import H
        low = 0
        # print(frequency[i]*4.135667696e-15)
        #low = i
        # integrand = conversion(1+z_injected[:],alpha=-3) / (1 + z_injected[:]) *  rate_per_volume_per_time[:] *energy_integral[i,:] #(1+z)**4 from splitting dln(1+z)=dz/(1+z)
        integrand = dlogz[0]/ H(z_injected[:]) / (1 + z_injected[:])** 3 *  rate_per_volume_per_time[:] *energy_integral[i,:] #(1+z)**4 from splitting dln(1+z)=dz/(1+z)
        # integrand = energy_integral[i,:] #(1+z)**4 from splitting dln(1+z)=dz/(1+z)
        # print(integrand) #(1+z)**4 from splitting dln(1+z)=dz/(1+z)
        # result[i] = trapz( integrand, z_injected)
        result2[i] = integrand.sum()
# result[i] = integrand.sum()

# result = np.empty_like( norm, dtype=np.float64 )
    # try:
    #     for i in range(len(normalization)):
    #         if normalization[i] != 0 and abs(result[i]) < np.inf :
    #             result[i] /= normalization[i]
    #         else:
    #             #result[i] = np.nan
    #             result[i] = 0.
    # except TypeError:
    #     if normalization != 0 and abs(result[i]) < np.inf :
    #         result[i] /= normalization
    #     else:
    #         #result[i] = np.nan
    #         result[i] = 0.

    # print(frequency,result,result2)
    # print(z_injected,result2)
    return result2/1e16*1e26
    # return result2

def spectral_distortions_finalize(frequency,spectral_distortions, **DarkOptions):
    u"""Prints the table of redshift and :math:`f_eff(z)` into :obj:`stdout`

    .. warning::
    For the correct usage of this package together with
    `CLASS <http://class-code.net>`_ the only allowed output
    are line with a single number, containing the number of the lines
    of the table to follow and the table

    +-------+-------+
    |   #nu  | EdNDE |
    +=======+=======+
    |   0   |       |
    +-------+-------+
    |  ...  |  ...  |
    +-------+-------+
    | 10000 |       |
    +-------+-------+

    Please make sure that all other message printed are silenced or
    at least covered by '#' (see :meth:`print_info <DarkAges.__init__.print_info>`)

    Parameters
    ----------
    spectral_distortions : :obj:`array-like`
    Array (:code:`shape = (k)`) with the values of the spectral distortions
    """

    #
    # sys.stdout.write(50*'#'+'\n')
    # sys.stdout.write('### This is the standardized output to be read by CLASS.\n### For the correct usage ensure that all other\n### "print(...)"-commands in your script are silenced.\n')
    # sys.stdout.write(50*'#'+'\n\n')
    # sys.stdout.write('#nu\t SI\n\n{:d}\n\n'.format( (last-first)))
    # # sys.stdout.write('{:.2e}\t{:.4e}\n'.format(min_nu,spectral_distortions[first]))
    # for idx in range(len(frequency)):
    # # for idx in range(first,last):
    #     sys.stdout.write('{:.5e}\t{:.4e}\n'.format(frequency[idx],spectral_distortions[idx]))
    # # sys.stdout.write('{:.5e}\t{:.4e}\n'.format(max_nu,spectral_distortions[last-1]))
    # f = open('DarkAgesModule/output_DarkAges_dist.tmp.dat','wt')
    # f = open(os.path.join(os.environ['DARKAGES_BASE'],'output_DarkAges_dist.tmp.dat'),'wt')

    sys.stdout.write(50*'#'+'\n')
    sys.stdout.write('### This is the standardized output to be read by CLASS.\n### For the correct usage ensure that all other\n### "print(...)"-commands in your script are silenced.\n')
    sys.stdout.write(50*'#'+'\n\n')
    # f.write(50*'#'+'\n')
    # f.write('### This is the standardized output to be read by CLASS.\n### For the correct usage ensure that all other\n### "print(...)"-commands in your script are silenced.\n')
    # f.write(50*'#'+'\n\n')
    # f.write('{:.2e}\t{:.4e}\n'.format(min_nu,spectral_distortions[first]))
    apply_smoothing = DarkOptions.get('apply_smoothing',False)
    if apply_smoothing is False:
        first = int(DarkOptions.get('first_index',1))
        last_idx = int(DarkOptions.get('last_index',0))

        last = len(frequency) - last_idx
        min_nu = DarkOptions.get('lower_E_bound',0.) ##to be updated
        max_nu = DarkOptions.get('upper_E_bound',5.01e6)
        # Define the number of bins
        sys.stdout.write('# 1:Frequency nu [GHz]     2:SD_tot\n\n{:d}\n\n'.format((last-first+1)))
        for idx in range(len(frequency)):
        # for idx in range(first,last):
            sys.stdout.write('{:.5e}\t{:.4e}\n'.format(frequency[idx],spectral_distortions[idx]))
        # f.write('{:.5e}\t{:.4e}\n'.format(max_nu,spectral_distortions[last-1]))
    else:
        original_bins = len(frequency)
        new_bins = int(DarkOptions.get('nbins_smoothing',100))
        bins_per_group = original_bins // new_bins
        freq_smooth = np.array([
            np.mean(frequency[i*bins_per_group:(i+1)*bins_per_group])
            for i in range(new_bins)
        ])
        dist_smooth = np.array([
            np.mean(spectral_distortions[i*bins_per_group:(i+1)*bins_per_group])
            for i in range(new_bins)
        ])
        # f.write('# 1:Frequency nu [GHz]     2:SD_tot\n\n{:d}\n\n'.format(len(freq_smooth)))
        # for idx in range(len(freq_smooth)):
        # # for idx in range(first,last):
        #     f.write('{:.5e}\t{:.4e}\n'.format(freq_smooth[idx],dist_smooth[idx]))
        # # f.write('{:.5e}\t{:.4e}\n'.format(max_nu,spectral_distortions[last-1]))
        sys.stdout.write('# 1:Frequency nu [GHz]     2:SD_tot\n\n{:d}\n\n'.format(len(freq_smooth)))
        for idx in range(len(freq_smooth)):
        # for idx in range(first,last):
            sys.stdout.write('{:.5e}\t{:.4e}\n'.format(freq_smooth[idx],dist_smooth[idx]))
        # f.write('{:.5e}\t{:.4e}\n'.format(max_nu,spectral_distortions[last-1]))
    # f.close()
