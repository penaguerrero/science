from __future__ import division
import numpy
#import math
import os
import logging
import collections
import string
from scipy import ndimage
from scipy import optimize
from pprint import pprint
from scipy import stats
import copy
from matplotlib import pyplot
#from uncertainties import unumpy
#from uncertainties import ufloat
#from numba import autojit

#@autojit
############################################################################################
# FITTING A CONTINUUM WITH SIGMA-CLIPPING

# The find_std function that determines the standard deviation is in the error functions section of this file

def sigma_clip_flux(flux_arr, sigmas_away):
    '''
    This function performs the clipping normalizing to sigma and according to the
    number of sigmas_away (i.e. the confidence level). It determines the standard
    deviation from the given array.
    FUNCTION RETURNS:
        1. standard deviation
        2. the sigma-clipped array
    '''
    std, _ = find_std(flux_arr)
    norm_flx = flux_arr /std
    clip_up = sigmas_away
    clip_down = (-1) * sigmas_away
    flx_clip = numpy.clip(norm_flx, clip_down, clip_up)
    return std, flx_clip*std

def iterate_sigma_clipdflx_usingMode(flx_arr, sigmas_away):
    '''
    This function does the iterations to converge to a standard deviation and then
    perform the sigma-clipping. If it cannot converge then it uses the flux mode as
    the standard deviation and then it clips.
    FUNCTION RETURNS:
        1. Standard deviation (or flux mode).
        2. Clipped flux array.
    '''
    normalize2 = 1e-14
    temp_f = flx_arr / normalize2
    decimals = 2
    rounded_f = numpy.around(temp_f, decimals)
    original_flux_mode = stats.mode(rounded_f, axis=None)
    for f in rounded_f:
        if f == max(rounded_f) or f == min(rounded_f):
            #print 'Found a max or min: ', f
            f = float(original_flux_mode[0])
    flux_mode = stats.mode(rounded_f, axis=None)
    #print 'flux_mode', flux_mode
    std = flux_mode[0] * normalize2
    #print 'sigma = ', std
    clip_up = sigmas_away
    clip_down = (-1) * sigmas_away
    i = 1
    end_loop = False
    # Prevent dividing by zero
    if std == 0.0:
        end_loop = True
        std =  numpy.median(flx_arr)
        #print 'Mode is zero, used median as std:', std
        n_arr = flx_arr / std
        flx_clip = numpy.clip(n_arr, clip_down, clip_up) * std
        return std, flx_clip
    # If the mode is not zero try to find standard deviation
    norm_flux = flx_arr / std
    flx_clip = numpy.clip(norm_flux, clip_down, clip_up) * std
    # Do iterations
    while end_loop == False:
        prev_std = std
        new_flx = flx_clip
        std, flx_clip = sigma_clip_flux(new_flx, sigmas_away)
        #print 'sigma = ', std
        # did it converge?
        std_diff = numpy.fabs(prev_std - std)
        #print 'std_diff = ', std_diff
        if std_diff == 0.0:
            end_loop = True
        # In case the function cannot converge, use mode flux as standard deviation
        elif (std_diff <= (flux_mode[0]*normalize2)) or (std == 0.0):
            end_loop = True
            std = float(flux_mode[0] * normalize2)         
            #print 'Did not converge, used flux mode as std: ', std
            n_arr = flx_arr / std
            flx_clip = numpy.clip(n_arr, clip_down, clip_up) * std
            #print 'std_diff', std_diff 
        i = i + 1
        # Stop the loop in case none of the other conditions are met
        if i > 1000:
            #print 'Reached maximum iterations without meeting the other conditions.'
            end_loop = True
    #print 'Number of iterations: ', i
    return std, flx_clip

def get_spec_sigclip(object_spec, window, sigmas_away):
    '''
    This function does the sigma-clip iterations over the given spectrum dividing it 
    in several windows.
    REQUIREMENTS:
        1. object_spec = 2D array of wavelength and flux.
        2. window = width of the windows in which to divide the spectrum.
        3. sigma_away = the desired confidence interval (1sigma=68.26895%, 2sigma=95.44997%, 
            3sigma=99.73002%, 4sigma=99.99366%, 5sigma=99.99994%)
    FUNCTION RETURNS:
        1. Full sigma-clipped 2D array of wavelengths and flux.
    '''
    # First window
    window_lo = object_spec[0][0]
    window_up, _ = find_nearest(object_spec[0], window_lo+window)
    print 'INITIAL Window: ', window_lo, window_up
    f_win = object_spec[1][(object_spec[0] <= window_up)]
    std, flx_clip = iterate_sigma_clipdflx_usingMode(f_win, sigmas_away)
    print 'sigma = ', std
    # List all the standard deviations and averages
    std_list = []
    avges_list = []
    avg_window = sum(flx_clip) / float(len(flx_clip))
    # Add the clipped fluxes
    clipd_fluxes_list = []
    for f in flx_clip:
        clipd_fluxes_list.append(f)
        std_list.append(std)
        avges_list.append(avg_window)
    # Following windows
    end_loop = False
    while end_loop == False:
        window_lo = window_up
        wup_increment = window_up + window
        # Make sure that the last window is not tiny
        dist2end = max(object_spec[0]) - window_up
        #print 'dist2end', dist2end
        if (wup_increment < max(object_spec[0])) and (dist2end >= window+100.0):
            window_up, _ = find_nearest(object_spec[0], wup_increment)
        else:
            end_loop = True
            window_up = max(object_spec[0])
        print 'Window: ', window_lo, window_up
        f_win = object_spec[1][(object_spec[0] > window_lo) & (object_spec[0] <= window_up)]
        std, flx_clip = iterate_sigma_clipdflx_usingMode(f_win, sigmas_away)
        print 'sigma = ', std
        avg_window = sum(flx_clip) / float(len(flx_clip))
        for f in flx_clip:
            clipd_fluxes_list.append(f)
            avges_list.append(avg_window)
            std_list.append(std)
    clipd_arr = numpy.array([object_spec[0], clipd_fluxes_list])
    std_arr = numpy.array([object_spec[0], std_list])
    avg_arr = numpy.array([object_spec[0], avges_list])
    return clipd_arr, std_arr, avg_arr

def find_mode_and_clip(flux_arr, threshold_fraction):
    # Find the mode in the rounded array -- just to make it easier
    normalize2 = 1e-15
    temp_f = flux_arr / normalize2
    decimals = 2
    rounded_f = numpy.around(temp_f, decimals)
    flux_mode = stats.mode(rounded_f, axis=None)
    # Make sure the mode is not zero
    if flux_mode[0] == 0.0:
        threshold = numpy.median(flux_arr) * normalize2
    else:
        threshold = float(flux_mode[0]) * normalize2
    #print 'Threshold = ', threshold
    # Normalize to the mode or median
    norm_flux = flux_arr / threshold
    clip_down = (-1) * threshold_fraction
    clip_up = threshold_fraction
    flux_clipd = numpy.clip(norm_flux, clip_down, clip_up)
    return flux_clipd * threshold

def clip_flux_using_modes(object_spec, window, threshold_fraction):
    '''
    This function does the mode-clip iterations over the given spectrum dividing it 
    in several windows.
    REQUIREMENTS:
        1. object_spec = 2D array of wavelength and flux.
        2. window = width of the windows in which to divide the spectrum.
        3. threshold_fraction = percentage of the flux mode to use for the clipping range.
    FUNCTION RETURNS:
        1. Full mode-clipped 2D array of wavelengths and flux.
    '''
    # First window
    window_lo = object_spec[0][0]
    window_up, _ = find_nearest(object_spec[0], window_lo+window)
    #print 'INITIAL Window: ', window_lo, window_up
    f_win = object_spec[1][(object_spec[0] <= window_up)]
    # Find the mode
    flux_clipd = find_mode_and_clip(f_win, threshold_fraction)
    # Add the clipped fluxes
    clipd_fluxes_list = []
    for f in flux_clipd:
        clipd_fluxes_list.append(f)
    # Following windows
    end_loop = False
    while end_loop == False:
        window_lo = window_up
        wup_increment = window_up + window
        # Make sure that the last window is not tiny
        dist2end = max(object_spec[0]) - window_up
        #print 'dist2end', dist2end
        if (wup_increment < max(object_spec[0])) and (dist2end >= window+100.0):
            window_up, _ = find_nearest(object_spec[0], wup_increment)
        else:
            end_loop = True
            window_up = max(object_spec[0])
        #print 'Window: ', window_lo, window_up
        f_win = object_spec[1][(object_spec[0] > window_lo) & (object_spec[0] <= window_up)]
        # Find the mode
        flux_clipd = find_mode_and_clip(f_win, threshold_fraction)
        for f in flux_clipd:
            clipd_fluxes_list.append(f)
    clipd_arr = numpy.array([object_spec[0], clipd_fluxes_list])
    return clipd_arr

def fit_continuum(object_name, object_spectra, z, sigmas_away=3.0, window=150, order=None, plot=True, z_correct=True, normalize=True, nullfirst150=True):
    '''
    This function shifts the object's data to the rest frame (z=0). The function then fits a 
    continuum to the entire spectrum, omitting the lines windows (it interpolates 
    in that region). It then CAN normalize the entire spectrum.
    The lines it looks for are those in the lines2fit.txt file.
    REQUIREMENTS:
    # object_spectra must be a 2D numpy array of wavelengths and fluxes
    # z is expected to be the redshift of the object
    # nth is the order of the polynomial, default is 5
    # thresold_fraction = freaction (percentage) to multiply the threshold. This modifies the width
                          of the flux band in which to allow interpolation of fluxes
    # window_width = the size of the spectrum window to be analyzed.
        * The default window size: 150 A but it can be set to take into account the whole spectrum.
    # nullfirst150 = do you want NOT to consider the first 150 Angstroms? Default=True, meaning no, 
                     do not take into account.
    FUNCTION RETURNS:
    # 2D numpy array of redshift-corrected wavenegths and fluxes.
    # 2D continuum numpy array of wavenegths and fluxes.
    '''
    print 'Calculating continuum...'
    # Bring the object to rest wavelength frame using 1+z = lambda_obs/lambda_theo - 1
    if z_correct == True:
        print '    *** Wavelengths corrected for redshift.'
        w_corr = object_spectra[0] / (1+float(z))
    else:
        print '    *** Wavelengths NOT YET corrected for redshift...'
        w_corr = object_spectra[0]
    # this is the array to find the continuum with
    corr_wf = numpy.array([w_corr, object_spectra[1]])
    wf, std_arr, avg_arr = get_spec_sigclip(corr_wf, window, sigmas_away)
    print 'numpy.shape(wf), numpy.shape(avg_arr)', numpy.shape(wf), numpy.shape(avg_arr)
    # Do you want to NOT take into account the first 150 angstroms and use the next window as average flux
    # but only if the window is small enough
    if window < 150.0:
        if nullfirst150:
            for i in range(len(avg_arr[1])):
                wl, idx = find_nearest(avg_arr[0], avg_arr[0][0]+150.0)
                avgfluxfirst150 = avg_arr[1][idx]
                if avg_arr[0][i] < wl:
                    avg_arr[1][i] = avgfluxfirst150
                print avg_arr[0][i], avg_arr[1][i]
    if order == None:
        fitted_continuum, nth, err_fit = get_best_polyfit(avg_arr, window)
        print 'order of best fit polynomial', nth
        print 'percentage of error of continuum fit = %0.2f' % err_fit
        poly_order = nth
    elif order != None:
        fitted_continuum = fit_polynomial(avg_arr, order)
        avg_chi2, chi2_list, window_cont_list, err_fit = find_reduced_chi2_of_polynomial(fitted_continuum, wf, window)
        print 'avg_chi2, chi2_list'
        print avg_chi2, chi2_list
        print 'percentage of error of continuum fit = %0.2f' % err_fit
        poly_order = order
        '''
        ###### USE THIS PART WHEN WANTING TO COMPARE WITH THE FLUX-MODE CLIPPING
        ### Alternatively, use the flux mode to clip
        mode_wf = clip_flux_using_modes(corr_wf, window, threshold_fraction=2.0)
        fitted_continuum_mode = fit_polynomial(mode_wf, order)    
        #print 'object_spectra[1][0], wf[1][0], fitted_continuum_mode[1][0]', object_spectra[1][0], wf[1][0], fitted_continuum_mode[1][0]
        '''
    # Plot if asked to
    if plot == True:
        pyplot.title(object_name)
        pyplot.suptitle('z-corrected spectra - polynomial of order = %s' % repr(poly_order))
        pyplot.xlabel('Wavelength [$\AA$]')
        pyplot.ylabel('Flux [ergs/s/cm$^2$/$\AA$]')    
        pyplot.plot(corr_wf[0], corr_wf[1], 'k', fitted_continuum[0], fitted_continuum[1], 'r')
        #pyplot.plot(wf[0], wf[1], 'b')    # trimed flux used to fit continuum
        #pyplot.plot(std_arr[0], std_arr[1], 'k--')    # average standard deviations
        pyplot.plot(avg_arr[0], avg_arr[1], 'y')    # average of trimed flux(blue)
        if order != None:
            #pyplot.plot(mode_wf[0], mode_wf[1], 'g--', fitted_continuum_mode[0],fitted_continuum_mode[1],'c--')
            for cont_arr in window_cont_list:
                pyplot.plot(cont_arr[0], cont_arr[1], 'y')
        pyplot.show()
        # Normalize to that continuum if norm=True
        print 'Continuum calculated. Normalization to continuum was set to: ', normalize
    if (normalize == True) and (plot == True):
        #before_norm = []
        norm_flux = numpy.array([])
        for i in range(len(corr_wf[1])):
            nf = numpy.abs(corr_wf[1][i]) / numpy.abs(fitted_continuum[1][i])
            if corr_wf[1][i] < 0.0:
                nf = -1 * nf
            print corr_wf[0][i], 'flux=', corr_wf[1][i], '   cont=',fitted_continuum[1][i], '   norm=', nf
            norm_flux = numpy.append(norm_flux, nf)
            #f = nf * numpy.abs(fitted_continuum[1][i])   # back to non-normalized fluxes
            #before_norm.append(f)
        norm_wf = numpy.array([wf[0], norm_flux])
        # Give the theoretical continuum for the line finding
        norm_continuum = theo_cont(corr_wf[0])
        pyplot.title(object_name)
        pyplot.suptitle('z-corrected spectra')
        pyplot.xlabel('Wavelength [$\AA$]')
        pyplot.ylabel('Normalized Flux')    
        pyplot.plot(norm_wf[0], norm_wf[1], 'b', norm_continuum[0], norm_continuum[1], 'r')
        pyplot.show()
        #pyplot.plot(norm_wf[0], before_norm, 'g')
        #pyplot.show()
        return norm_wf, norm_continuum, err_fit
    else:
        return corr_wf, fitted_continuum, err_fit

def write_wavflxcont_file(wavelengths, fluxes, continuum, file_name):
    ''' This function is used when wanting to write a file containing only 3 columns: wavelengths, fluxes, and fitted coninuum. '''
    txt = open(file_name, '+w')
    print >> txt, '{:<14} {:<30} {:<30}'.format('Wavelength [A]', 'Flux [ergs/s/cm$^2$/$\AA$]', 'Continuum [ergs/s/cm$^2$/$\AA$]')
    for w, f, c in zip(wavelengths, fluxes, continuum):
        print >> txt, '{:<4.10f} {:<20.10e} {:<20.10e}'.format(w, f, c)
    txt.close()

def get_best_polyfit(continuum_arr, window):
    order_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    pol_fit_list = []
    avgchi2_list = []
    for order in order_list:
        pol_fit = fit_polynomial(continuum_arr, order)
        pol_fit_list.append(pol_fit)
        avg_chi2, _, _, err_fit = find_reduced_chi2_of_polynomial(pol_fit, continuum_arr, window)
        avgchi2_list.append(avg_chi2)
    best_fit = min(avgchi2_list)
    best_fit_idx = find_index_in_list(avgchi2_list, best_fit)
    print 'These are the average reduced Chi2 values: '
    print avgchi2_list
    return pol_fit_list[best_fit_idx], order_list[best_fit_idx], err_fit

def fit_polynomial(arr, order):
    '''
    This function fits a polynomial to the given array.
    arr = is 2D array of wavelengths and fluxes
    RETURNS:
    the 2D array of the fitted polynomial.
    '''
    # Polynolial of the form y = Ax^5 + Bx^4 + Cx^3 + Dx^2 + Ex + F
    coefficients = numpy.polyfit(arr[0], arr[1], order)
    polynomial = numpy.poly1d(coefficients)
    f_pol = polynomial(arr[0])
    fitted_poly = numpy.array([arr[0], f_pol])
    return fitted_poly

def find_reduced_chi2_of_polynomial(polynomialfit_arr, continuum_arr, window):
    ''' Determine if the adjusted polynomial is a good enough fit to the continuum fluxes.
    1. Null hypothesis = there are equal number of points above and below the fitted polynomial (it passes through the middle of the data).
    2. We expect 50% of the points above the continuum and 50% below -- the expected value is then the standard deviation that I 
        used to define the continuum range fluxes.
    3. The observed values are the polynomial fluxes fitted to the trimed flux array that I defined as the continuum.
    4. The expected values will be the mean values of the window fluxes of the continuum array.
    5. Degrees of freedom, df = 1 (in this case there are 2 outcomes, above and below the continuum, so  df = outcomes - 1).
    '''
    # First window
    window_lo = polynomialfit_arr[0][0]
    window_up, _ = find_nearest(polynomialfit_arr[0], window_lo+window)
    #print 'INITIAL Window: ', window_lo, window_up
    pol_wavs_lists = []
    pol_flxs_lists = []
    w_win = polynomialfit_arr[0][(polynomialfit_arr[0] <= window_up)]
    f_win = polynomialfit_arr[1][(polynomialfit_arr[0] <= window_up)]
    pol_wavs_lists.append(w_win)
    pol_flxs_lists.append(f_win)
    # Define per window the expected_value_arr as the median of the trimed flux that I defined as continuum_arr
    window_cont_list = []
    w_cont = continuum_arr[0][(continuum_arr[0] <= window_up)]
    f_cont = continuum_arr[1][(continuum_arr[0] <= window_up)]
    window_cont_mean = sum(f_cont) / float(len(f_cont))
    wc = []
    for _ in w_cont:
        wc.append(window_cont_mean)
    window_continuum_mean_arr = numpy.array([w_cont, wc])
    window_cont_list.append(window_continuum_mean_arr)
    all_local_err_fit = []
    # Approximate the error in the continuum fitting by number of points in window
    local_err_fit = 1 / ((float(len(f_win)))**0.5)
    all_local_err_fit.append(local_err_fit)
    # Following windows
    end_loop = False
    while end_loop == False:
        window_lo = window_up
        wup_increment = window_up + window
        # Make sure that the last window is not tiny
        dist2end = max(polynomialfit_arr[0]) - window_up
        #print 'dist2end', dist2end
        if (wup_increment < max(polynomialfit_arr[0])) and (dist2end >= window+100.0):
            window_up, _ = find_nearest(polynomialfit_arr[0], wup_increment)
        else:
            end_loop = True
            window_up = max(polynomialfit_arr[0])
        #print 'Window: ', window_lo, window_up
        w_win = polynomialfit_arr[0][(polynomialfit_arr[0] > window_lo) & (polynomialfit_arr[0] <= window_up)]
        f_win = polynomialfit_arr[1][(polynomialfit_arr[0] > window_lo) & (polynomialfit_arr[0] <= window_up)]
        pol_wavs_lists.append(w_win)
        pol_flxs_lists.append(f_win)
        # Now the window median of the continuum
        w_cont = continuum_arr[0][(continuum_arr[0] > window_lo) & (continuum_arr[0] <= window_up)]
        f_cont = continuum_arr[1][(continuum_arr[0] > window_lo) & (continuum_arr[0] <= window_up)]
        window_cont_mean = sum(f_cont) / float(len(f_cont))
        wc = []
        for _ in w_cont:
            wc.append(window_cont_mean)
        window_continuum_mean_arr = numpy.array([w_cont, wc])
        window_cont_list.append(window_continuum_mean_arr)
        # Approximate the error in the continuum fitting by number of points in window
        local_err_fit = 1 / ((float(len(f_win)))**0.5)
        all_local_err_fit.append(local_err_fit)
    err_fit = 1 / sum(all_local_err_fit)
    chi2_list = []
    for f_pol, cont in zip(pol_flxs_lists, window_cont_list):
        degreees_freedom = float(len(f_pol)) - 1
        # Divide the polynomial array into the same binning as the windows
        chi_sq = find_chi2(f_pol, cont[1])
        #print 'degreees_freedom = ', degreees_freedom
        reduced_chi_sq = chi_sq / degreees_freedom
        chi2_list.append(reduced_chi_sq)
    avg_chi2 = sum(chi2_list) / float(len(chi2_list))
    return avg_chi2, chi2_list, window_cont_list, err_fit

def find_chi2(observed_arr, expected_arr):
    mean = sum(observed_arr) / float(len(observed_arr))
    variance_list = []
    for o in observed_arr:
        variance = (o - mean)**2
        variance_list.append(variance)
    variance =  sum(variance_list) / float(len(observed_arr))
    #print 'variance', variance
    chi2 = []
    for o, e in zip(observed_arr, expected_arr):
        diff_squared = (o - e)**2
        chi2.append(diff_squared) 
    chi_squared = sum(chi2) / variance
    return chi_squared

############################################################################################
# GATHER TEXT FILES OF SPECTRA INTO A SINGLE FILE
def gather_specs(text_file_list, name_out_file, reject=0.0, start_w=None, create_txt=True, err_cont_fit=None, errs_files=None):
    '''
    this function gathers all the text files into a single text file.
    # specs_list = list of the text files to be gathered
    # name_out_file = name of the output file with the gathered wavelengths and fluxes
    # reject = limit of the angstroms to be rejected from the edges of the detector (default is set to 0.0=No rejections)
    # start_w = will force the function to start the accepted array from this wavelength
    # create_txt = if set to False the output will be just the full data into a list of lists
    # errs_files is the list of error text files
    RETURNS:
    # the text file with the gathered data
    '''
    accepted_catalog_wavelength = []
    accepted_observed_wavelength = [] 
    accepted_element = []
    accepted_ion =[]
    accepted_forbidden = []
    accepted_how_forbidden = []
    accepted_width = []
    accepted_flux = []
    accepted_continuum = []
    accepted_EW = []
    all_err_fit = []
    flux_errs = []
    cont_errs = []
    ew_errs = []
    for spec in text_file_list:
        # read the file
        cols_in_file = readlines_from_lineinfo(spec)
        if err_cont_fit != None:
            err_cont_fit = get_err_cont_fit(spec)
            all_err_fit.append(err_cont_fit)
        if errs_files != None:
            # We are only interested in columns of: flux error, continuum error, and EW error
            if 'nuv' in spec:
                efile = errs_files[0]
            elif 'opt' in spec:
                efile = errs_files[1]
            elif 'nir' in spec:
                efile = errs_files[2]
            eff, efc, efe = numpy.loadtxt(efile, skiprows=1, usecols=(2,5,8), unpack=True)
        # Each spec contains the following:
        # 0=catalog_wavelength, 1=observed_wavelength, 2=element, 3=ion, 4=forbidden, 5=how_forbidden, 6=width, 7=flux, 8=continuum, 9=EW
        # These are lists, make the observed wavelengths a numpy array to choose the accepted range of data
        obs_wavs = numpy.array(cols_in_file[1])
        if start_w == None:
            ini_wav = obs_wavs[0]+reject
        else:
            # Make sure to start the right array and leave the others with the regular reject
            if start_w <= 2000.0:
                if 'nuv' in spec:
                    ini_wav = start_w
                else:
                    ini_wav = obs_wavs[0]+reject                
            if (start_w > 2000.0) and (start_w < 5000.0):
                if 'opt' in spec:
                    ini_wav = start_w
                else:
                    ini_wav = obs_wavs[0]+reject                
            if start_w >= 5000.0:
                if 'nir' in spec:
                    ini_wav = start_w
                else:
                    ini_wav = obs_wavs[0]+reject                
        initial_wav, initial_idx = find_nearest(obs_wavs, ini_wav)
        ending_wav, ending_idx = find_nearest(obs_wavs, obs_wavs[-1]+reject)
        print 'Data from %s will start at %f' % (spec, initial_wav)
        print 'and will end at %f' % (ending_wav)
        # Select the data by slicing
        for i in range(len(cols_in_file[0][initial_idx : ending_idx])):
            idx = initial_idx + i 
            accepted_catalog_wavelength.append(cols_in_file[0][idx])
            accepted_observed_wavelength.append(cols_in_file[1][idx])
            accepted_element.append(cols_in_file[2][idx])
            accepted_ion.append(cols_in_file[3][idx])
            accepted_forbidden.append(cols_in_file[4][idx])
            accepted_how_forbidden.append(cols_in_file[5][idx])
            accepted_width.append(cols_in_file[6][idx])
            accepted_flux.append(cols_in_file[7][idx])
            accepted_continuum.append(cols_in_file[8][idx])
            accepted_EW.append(cols_in_file[9][idx])
            if errs_files != None:
                flux_errs.append(eff[idx])
                cont_errs.append(efc[idx])
                ew_errs.append(efe[idx])
    accepted_cols_in_file = [accepted_catalog_wavelength, accepted_observed_wavelength, accepted_element, accepted_ion, accepted_forbidden, 
                             accepted_how_forbidden, accepted_width, accepted_flux, accepted_continuum, accepted_EW]
    if errs_files != None:
        accepted_errs = [flux_errs, ew_errs, cont_errs]
    # Choose the right intensities if lines are repeated
    # There are no repetitions between the NUV and Opt but there could be a few from the Opt to the NIR. 
    # Because the sensibility of the detector is better with the g750l than with the g430l, we want to keep NIR the lines.
    repeated_lines = collections.Counter(accepted_catalog_wavelength)
    rl_list = [i for i in repeated_lines if repeated_lines[i]>1]
    for rl in rl_list:
        idx_of_line_to_be_removed = find_index_in_list(accepted_catalog_wavelength, rl)
        if (rl > 5000.0) and (rl < 5600.0):
            for each_list in accepted_cols_in_file:
                # Since the line of the g430l filter will be the first occurence, remove that
                each_list.pop(idx_of_line_to_be_removed)
            if errs_files != None:
                for each_list in accepted_errs:
                    each_list.pop(idx_of_line_to_be_removed)
    # Create the table of Net Fluxes and EQWs
    if create_txt == True:
        txt_file = open(name_out_file, 'w+')
        print >> txt_file,  '# Redshift-corrected lines'
        print >> txt_file,   '# Positive EW = emission        Negative EW = absorption' 
        if err_cont_fit != None:        
            print >> txt_file,   '# Percentage Errors of Continuum Fits: NUV, Opt, NIR = %0.2f, %0.2f, %0.2f' % (all_err_fit[0], all_err_fit[1], all_err_fit[2])
        else:
            print >> txt_file,   '#'
        print >> txt_file,   '#    NUV: wav <= 2000,   Opt: 2000 > wav < 5000,   NIR: wav >= 5000'
        if errs_files != None:
            print >> txt_file,  ('{:<12} {:<12} {:>10} {:<4} {:<9} {:>8} {:<9} {:<12} {:<10} {:<7} {:<12} {:<7} {:<6} {:<6}'.format('# Catalog WL', 'Observed WL', 'Element', 'Ion', 'Forbidden', 'HowForb', 'Width[A]', 'Flux [cgs]', 'FluxErr', '%Err', 'Continuum [cgs]', 'EW [A]', 'EWErr', '%Err'))
            for cw, w, e, i, fd, h, s, F, Fe, C, ew, ewe in zip(accepted_cols_in_file[0], accepted_cols_in_file[1], accepted_cols_in_file[2], accepted_cols_in_file[3], accepted_cols_in_file[4], 
                                                       accepted_cols_in_file[5], accepted_cols_in_file[6], accepted_cols_in_file[7], accepted_errs[0], accepted_cols_in_file[8], accepted_cols_in_file[9], accepted_errs[1]):
                #print cw, w, e, i, fd, h, s, F, Fe, C, ew, ewe
                Fep = (Fe * 100.) / numpy.abs(F)
                ewep = (ewe * 100.) / numpy.abs(ew)
                print >> txt_file,  ('{:<12.3f} {:<12.3f} {:>10} {:<6} {:<8} {:<8} {:<6} {:>12.3e} {:>10.3e} {:>6.1f} {:>13.3e} {:>10.3f} {:>6.3f} {:>6.1f}'.format(cw, w, e, i, fd, h, s, F, Fe, Fep, C, ew, ewe, ewep))
        else:
            print >> txt_file,  ('{:<12} {:<12} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16} {:>16} {:>12}'.format('# Catalog WL', 'Observed WL', 'Element', 'Ion', 'Forbidden', 'How much', 'Width[A]', 'Flux [cgs]', 'Continuum [cgs]', 'EW [A]'))
            for cw, w, e, i, fd, h, s, F, C, ew in zip(accepted_cols_in_file[0], accepted_cols_in_file[1], accepted_cols_in_file[2], accepted_cols_in_file[3], accepted_cols_in_file[4], 
                                                       accepted_cols_in_file[5], accepted_cols_in_file[6], accepted_cols_in_file[7], accepted_cols_in_file[8], accepted_cols_in_file[9]):
                print >> txt_file,  ('{:<12.3f} {:<12.3f} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16.3e} {:>16.3e} {:>12.3f}'.format(cw, w, e, i, fd, h, s, F, C, ew))
        txt_file.close()
        print 'File   %s   writen!' % name_out_file
    if err_cont_fit != None:
        return accepted_cols_in_file, accepted_errs, all_err_fit
    else:
        return accepted_cols_in_file, accepted_errs
    
def get_err_cont_fit(text_file):
    f = open(text_file, 'r')
    list_rows_of_file = f.readlines()
    f.close()
    for row in list_rows_of_file:
        if 'Percentage Error' in row:
            line_string_list = string.split(row, sep='=')
            err_cont_fit = float(line_string_list[1])
    return err_cont_fit

############################################################################################
# LINE INFORMATION
def get_obj_files2use(object_name, specs, add_str=None):
    if add_str != None:
        nuv = object_name+add_str+"_nuv.txt"
        opt = object_name+add_str+"_opt.txt"
        nir = object_name+add_str+"_nir.txt"
    else:
        nuv = object_name+"_nuv.txt"
        opt = object_name+"_opt.txt"
        nir = object_name+"_nir.txt"
    full_file_list = [nuv, opt, nir]
    # Determine what files to use
    text_file_list = []
    for item in specs:
        tf = full_file_list[item]
        text_file_list.append(tf)        
    return text_file_list, full_file_list

def loadtxt_from_files(object_name, add_str, specs, text_files_path):
    text_file_list, full_file_list = get_obj_files2use(object_name, specs, add_str)
    # List of the data contained in each file
    data = []
    for i in range(len(text_file_list)):
        txtfile_path = os.path.join(text_files_path, text_file_list[i])
        print 'Opening: ', txtfile_path
        # The file is expected to be two columns without header: wavelengths, fluxes
        data_file = numpy.loadtxt(txtfile_path, unpack=True)
        #print 'LIMITS', data_file[0][0], max(data_file[0])
        data.append(data_file)
    return data, full_file_list

def readlines_from_lineinfo(text_file, cols_in_file=None):
    if cols_in_file == None:
        catalog_wavelength = []
        observed_wavelength = [] 
        element = []
        ion =[]
        forbidden = []
        how_forbidden = []
        width = []
        flux = []
        continuum = []
        EW = []
        cols_in_file = [catalog_wavelength, observed_wavelength, element, ion, forbidden, how_forbidden, width, flux, continuum, EW]
    # List of the data contained in the file
    if type(text_file) is list:
        for each_file in text_file:
            f = open(each_file, 'r')
            list_rows_of_file = f.readlines()
            f.close()
    else:
        f = open(text_file, 'r')
        list_rows_of_file = f.readlines()
        f.close()
    widths_faintObj = []
    widths_strongObj = []
    for row in list_rows_of_file:
        # Disregard comment symbol
        if '#'  not in row:
            # Split each row into columns
            line_data = row.split()
            # append the element into each column in the cols_in_file
            for item, col in zip(line_data, cols_in_file):
                if '.' in item:
                    item = float(item)
                col.append(item)
        if 'widths' in row:
            kk = string.split(row, '=')
            if 'faint' in kk[0]:
                kk2 = string.split(kk[1], sep=',')
                for number in kk2:
                    widths_faintObj.append(float(number))
            elif 'strong' in kk[0]:
                kk2 = string.split(kk[1], sep=',')
                for number in kk2:
                    widths_strongObj.append(float(number))
    if len(widths_faintObj) > 0:
        return cols_in_file, widths_faintObj, widths_strongObj
    else:
        return cols_in_file

def n4airvac_conversion(wav):
    '''This function finds the index of refraction for that wavelength.
        *** Took the equation from IAU convention for wavelength conversion described in 
            Morton (1991, ApJS, 77, 119)'''
    sigma_wav = 10000/wav
    n = 1 + 6.4328e-5 + 2.94981e-2/(146*sigma_wav*sigma_wav) + 2.5540e-4/(41*sigma_wav*sigma_wav)
    return n

def find_lines_info(object_spectra, continuum, Halpha_width, text_table=False, vacuum=False, faintObj=False, linesinfo_file_name=None, do_errs=None):
    '''
    This function takes the object and continuum arrays to find the
    lines given in the lines_catalog.txt file.
    *** WARNING:
        This function assumes that object_spectra has already been 
        corrected for redshift.
    REQUIREMENTS:
    # object_spectra must be a 2D numpy array of wavelengths and fluxes
    # continuum must be a 2D numpy array of wavelengths and fluxes
    # vacuum allows to choose to use either air (vacuum=False) or vacuum wavelengths (vacuum=True)
    # n = index of refraction
        * Default n=0.999271 was taken from NIST, 2013.
    FUNCTION RETURNS:
    # catalog_wavs_found = the list of lines it found from the catalog.
    # central_wavelength_list = the list of lines it found in the object spectra
    # width_list = the list of widths of the lines
    # net_fluxes_list = the sum of the fluxes over the line width
    # continuum_list = the average continuum value over the line width
    # EWs_list = the list of EWs it calculated
    # if text_table=True a text file containing all this information
    # do_errs = err_instrument, err_continuum  -- this is a list of two lists containing the instrument and continuum errors
    '''
    # Read the line_catalog file, assuming that the path is the same:
    # '/Users/name_of_home_directory/Documents/AptanaStudio3/science/science/spectrum/lines_catalog.txt'
    line_catalog_path = os.path.abspath('../../science/science/spectrum/lines_catalog.txt')
    # Define the columns of the file
    wavelength = []
    element = []
    ion =[]
    forbidden = []
    how_forbidden = []
    transition = []
    strong_line = []
    cols_in_file = [wavelength, element, ion, forbidden, how_forbidden, transition, strong_line]
    # Define the list of the files to be read
    text_file_list = [line_catalog_path]
    # Read the files
    data, widths_faintObj, widths_strongObj = readlines_from_lineinfo(text_file_list, cols_in_file)
    wavelength, element, ion, forbidden, how_forbidden, transition, strong_line = data
    # If the wavelength is grater than 2000 correct the theoretical air wavelengths to vacuum using the IAU
    # standard for conversion from air to vacuum wavelengths is given in Morton (1991, ApJS, 77, 119). To
    # correct find the refraction index for that wavelength and then use:
    #       wav_vac / wav_air -1 = n - 1
    # (To start I usded NIST, n=0.999271)
    wavs_air = []
    wavs_vacuum = []
    for w in wavelength:
        # separate air and vacuum wavelengths into 2 lists
        if w < 2000.0:
            # For keeping all vacuum wavelengths
            wavs_vacuum.append(w)
            # For converting vaccuum to air
            #print 'Wavelength < 2000, converting to air'
            wav_refraction_index = n4airvac_conversion(w)
            #print 'Refraction index  n = %f' % (wav_refraction_index)
            wair = w / (2 - wav_refraction_index)
            wavs_air.append(wair)
        elif w >= 2000.0:
            # For converting to vacuum wavelengths
            wav_refraction_index = n4airvac_conversion(w)
            wvac = w * (2 - wav_refraction_index)
            wavs_vacuum.append(wvac)
            # For keeping all AIR wavelengths
            #print 'Wavelength > 2000, keeping air'
            wavs_air.append(w)
            
    # Determine the strength of the lines: no_and_weak(nw)=5A, no(with respect to strong lines)=7, weak=10, medium=15, yes=25, super=35
    width = []
    for sline in strong_line:
        if faintObj == True: 
            if sline == "nw":
                s = widths_faintObj[0]
            elif sline == "no":
                s = widths_faintObj[1]
            elif sline == "weak":
                s = widths_faintObj[2]
            elif sline == "medium":
                s = widths_faintObj[3]
            elif sline == "yes":
                s = widths_faintObj[4]
            elif sline == "super":
                s = widths_faintObj[5]
            elif sline == "Halpha":
                s = Halpha_width
            width.append(s)
        else:
            if sline == "nw":
                s = widths_strongObj[0]
            elif sline == "no":
                s = widths_strongObj[1]
            elif sline == "weak":
                s = widths_strongObj[2]
            elif sline == "medium":
                s = widths_strongObj[3]
            elif sline == "yes":
                s = widths_strongObj[4]
            elif sline == "super":
                s = widths_strongObj[5]
            elif sline == "Halpha":
                s = Halpha_width
            width.append(s)
    # Search in the object given for the lines in the lines_catalog
    lines_catalog = (wavs_air, wavs_vacuum, element, ion, forbidden, how_forbidden, transition, width)
    net_fluxes_list = []
    EWs_list = []
    central_wavelength_list =[]
    catalog_wavs_found = []
    continuum_list =[]
    width_list = []
    found_element = []
    found_ion = []
    found_ion_forbidden = []
    found_ion_how_forbidden = []
    errs_net_fluxes = []
    errs_ews = [] 
    # but choose the right wavelength column
    if vacuum == True:
        use_wavs = 1
        use_wavs_text = '# Used VACUUM wavelengths to find lines from line_catalog.txt'
    if vacuum == False:
        use_wavs = 0
        use_wavs_text = '# Used AIR wavelengths to find lines from line_catalog.txt'
    print 'vacuum was set to %s, %s' % (vacuum, use_wavs_text)
    if do_errs != None:
        err_instrument, err_continuum = do_errs
        perc_err_continuum = err_continuum*100.0
        err_lists = get_flux_cont_errs(object_spectra, continuum, err_instrument, err_continuum)
        # err_lists contains err_fluxes, err_contfl 
    for i in range(len(lines_catalog[0])):
        # find the line in the catalog that is closest to a 
        line_looked_for = lines_catalog[use_wavs][i]
        nearest2line = find_nearest_within(object_spectra[0], line_looked_for, 10.0)
        if nearest2line > 0.0:  
            catalog_wavs_found.append(line_looked_for)
            # If the line is in the object spectra, measure the intensity and equivalent width
            # according to the strength of the line
            central_wavelength = object_spectra[0][(object_spectra[0] == nearest2line)]
            line_width = lines_catalog[7][i]
            round_line_looked_for = numpy.round(line_looked_for, decimals=0)
            if (round_line_looked_for == 1907.0) or (round_line_looked_for == 1909.0):
                if faintObj:
                    line_width = 3.0
                else:
                    line_width = 7.5
            if (line_looked_for ==  4267.15) or (line_looked_for == 4640.0) or (line_looked_for == 4650.0):
                line_width = 5.0 
            lower_wav = central_wavelength - (line_width/2)
            upper_wav = central_wavelength + (line_width/2)
            if do_errs != None:
                F, C, err_F, ew, lolim, uplim, err_ew = get_net_fluxes(object_spectra, continuum, line_looked_for, lower_wav, upper_wav, do_errs=err_lists)
                errs_net_fluxes.append(err_F)
                errs_ews.append(err_ew)
            else:
                F, C, ew, lolim, uplim = get_net_fluxes(object_spectra, continuum, line_looked_for, lower_wav, upper_wav)   
            final_width = float(uplim - lolim)
            final_width = numpy.round(final_width, decimals=1)
            central_wavelength = float((uplim+lolim)/2.0)
            print '\n Looking for ',  round_line_looked_for #***
            print 'This is the closest wavelength in the data to the target line: ', nearest2line
            print 'center=', central_wavelength,'  initial_width=',line_width, '  final_width = %f' % final_width, '    ew=', ew
            print 'center=', central_wavelength,'  Flux=',F, '  ew=', ew, '  from ', lolim, '  to ', uplim
            #if (round_line_looked_for == 1907.0) or (round_line_looked_for == 1909.0):
            #    raw_input()
            #if (round_line_looked_for == 5007):
            #    raw_input()
            #if line_looked_for >= 5260.0:
                #line_wavs = object_spectra[0][(object_spectra[0] >= lolim) & (object_spectra[0] <= uplim)]
                #line_flxs = object_spectra[1][(object_spectra[0] >= lolim) & (object_spectra[0] <= uplim)]
                #deblend_line(line_wavs, line_flxs, final_width, C, F, ew, plot_fit=True)
            #    raw_input()
            width_list.append(final_width)
            central_wavelength_list.append(central_wavelength)
            continuum_list.append(C)
            net_fluxes_list.append(F)
            EWs_list.append(ew) 
            found_element.append(lines_catalog[2][i])
            found_ion.append(lines_catalog[3][i])
            found_ion_forbidden.append(lines_catalog[4][i])
            found_ion_how_forbidden.append(lines_catalog[5][i])
    # Create the table of Net Fluxes and EQWs
    if linesinfo_file_name != None:
        if text_table == True:
            #linesinfo_file_name = raw_input('Please type name of the .txt file containing the line info. Use the full path.')
            txt_file = open(linesinfo_file_name, 'w+')
            print >> txt_file,  use_wavs_text
            print >> txt_file,   '# Positive EW = emission        Negative EW = absorption' 
            if do_errs != None:
                print >> txt_file,   '# Percentage Error of Continuum Fit = %0.2f' % perc_err_continuum
            else:
                print >> txt_file,   '#'
            print >> txt_file,  ('{:<12} {:<12} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16} {:>16} {:>12}'.format('# Catalog WL', 'Observed WL', 'Element', 'Ion', 'Forbidden', 'How much', 'Width[A]', 'Flux [cgs]', 'Continuum [cgs]', 'EW [A]'))
            for cw, w, e, i, fd, h, s, F, C, ew in zip(catalog_wavs_found, central_wavelength_list, found_element, found_ion, found_ion_forbidden, found_ion_how_forbidden, width_list, net_fluxes_list, continuum_list, EWs_list):
                #print 'cw:',type(cw), 'w:',type(w), 'e:',type(e), 'i:',type(i), 'fd:',type(fd), 'h:',type(h), 's:',type(s), 'F:',type(F), 'C:',type(C), 'ew:',type(ew)
                print >> txt_file,  ('{:<12.3f} {:<12.3f} {:>12} {:<12} {:<12} {:<12} {:<10.2f} {:>16.3e} {:>16.3e} {:>12.3f}'.format(cw, w, e, i, fd, h, s, F, C, ew))
            txt_file.close()
            print 'File   %s   writen!' % linesinfo_file_name
        elif text_table == False:
            print '# Positive EW = emission        Negative EW = absorption' 
            print ('{:<12} {:<12} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16} {:>16} {:>12}'.format('# Catalog WL', 'Observed WL', 'Element', 'Ion', 'Forbidden', 'How much', 'Width[A]', 'Flux [cgs]', 'Continuum [cgs]', 'EW [A]'))
            for cw, w, e, i, fd, h, s, F, C, ew in zip(catalog_wavs_found, central_wavelength_list, found_element, found_ion, found_ion_forbidden, found_ion_how_forbidden, width_list, net_fluxes_list, continuum_list, EWs_list):
                print ('{:<12.3f} {:<12.3f} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16.3e} {:>16.3e} {:>12.3f}'.format(cw, w, e, i, fd, h, s, F, C, ew))
        if do_errs != None:
            return catalog_wavs_found, central_wavelength_list, width_list, net_fluxes_list, continuum_list, EWs_list, errs_net_fluxes, errs_ews
        else:
            return catalog_wavs_found, central_wavelength_list, width_list, net_fluxes_list, continuum_list, EWs_list
    else:
        if do_errs != None:
            return catalog_wavs_found, central_wavelength_list, width_list, net_fluxes_list, continuum_list, EWs_list, errs_net_fluxes, errs_ews
        else:
            return catalog_wavs_found, central_wavelength_list, width_list, net_fluxes_list, continuum_list, EWs_list

def get_flux_cont_errs(object_spectra, continuum, err_instrument, err_continuum):
    # Determine the absolute error and get the arrays of plus and minus fluxes
    err_fluxes = []
    err_contfl = []
    for f, c in zip(object_spectra[1], continuum[1]):
        ferr = f * err_instrument
        err_fluxes.append(ferr)
        cerr = c * err_continuum
        err_contfl.append(cerr)
    return err_fluxes, err_contfl

def get_lineinfo_uncertainties(object_spectra, continuum, Halpha_width, faintObj, err_instrument, err_continuum):
    # Determine the absolute error and get the arrays of plus and minus fluxes
    err_fluxes_plus = []
    err_fluxes_minus = []
    err_contfl_plus = []
    err_contfl_minus = []
    wavelengths = []
    for w, f, c in zip(object_spectra[0], object_spectra[1], continuum[1]):
        wavelengths.append(w)
        ferr = f * err_instrument
        fplus = f + ferr
        fminus = f - ferr
        err_fluxes_plus.append(fplus)
        err_fluxes_minus.append(fminus)
        cerr = c * err_continuum
        cplus = c + cerr
        cminus = c - cerr
        err_contfl_plus.append(cplus)
        err_contfl_minus.append(cminus)
    object_spectra_plus = numpy.array([wavelengths, err_fluxes_plus]) 
    object_spectra_minus = numpy.array([wavelengths, err_fluxes_minus]) 
    continuum_plus = numpy.array([wavelengths, err_contfl_plus])
    continuum_minus = numpy.array([wavelengths, err_contfl_minus])
    # Now run the plus and minus fluses through the lineinfo function
    object_lines_info_plus = find_lines_info(object_spectra_plus, continuum_plus, Halpha_width=Halpha_width, text_table=False,  
                                             vacuum=False, faintObj=faintObj, linesinfo_file_name=None, do_errs=None)
    object_lines_info_minus = find_lines_info(object_spectra_minus, continuum_minus, Halpha_width=Halpha_width, text_table=False,  
                                              vacuum=False, faintObj=faintObj, linesinfo_file_name=None, do_errs=None)
    # get the final error by assuming that it is symetric: (x_plus - x_minus)/2
    # line_info contains: catalog_wavs_found, central_wavelength_list, width_list, net_fluxes_list, continuum_list, EWs_list
    _, _, _, net_fluxes_list_plus, continuum_list_plus, EWs_list_plus = object_lines_info_plus
    _, _, _, net_fluxes_list_minus, continuum_list_minus, EWs_list_minus = object_lines_info_minus
    err_fluxes = []
    err_continuum = []
    err_ews = []
    for nfp, clp, ewp, nfm, clm, ewm in zip(net_fluxes_list_plus, continuum_list_plus, EWs_list_plus, net_fluxes_list_minus, continuum_list_minus, EWs_list_minus):
        e_flx = (numpy.abs(nfp) - numpy.abs(nfm)) / 2.0
        e_cont = (numpy.abs(clp) - numpy.abs(clm)) / 2.0
        e_ew = (numpy.abs(ewp) - numpy.abs(ewm)) / 2.0
        err_fluxes.append(numpy.abs(e_flx))
        err_continuum.append(numpy.abs(e_cont))
        err_ews.append(numpy.abs(e_ew))
    return err_fluxes, err_continuum, err_ews

def get_net_fluxes(object_spectra, continuum, line_looked_for, lower_wav, upper_wav, do_errs=None):
    '''
    This function finds the integrated flux of the line given by the lower and upper
    wavelengths, along with the continuum value, and the equivalent width.
    REQUIREMENTS:
    # object_spectra = the 2D array of wavelength and flux
    # continuum = the 2D array of wavelength and flux for the continuum
    # line_looked_for = the target line that we want to measure 
    # lower_wav, upper_wav = limits of integration
    FUNCTION RETURNS:
    # the net flux and corresponding continuum of the integration between lower and upper wavelengths
    # upper and lower limits of the integration
    # equivalent with
    # continuum value
    # (only if asked for) errors in the flux and EW 
    '''
    net_continua_f = continuum[1][(continuum[0] >= lower_wav) & (continuum[0] <= upper_wav)]
    #print 'len(net_continua)', len(net_continua_F)
    #elements = 10
    #_, _, _, net_continua_f = fill_EWarr(object_spectra, continuum, lower_wav, upper_wav, elements)
    C = midpoint(net_continua_f[0], net_continua_f[-1])
    # simple equivalent width routine
    if do_errs != None:
        #print 'do_errs != None'
        #ew, lower_wav, upper_wav, err_ew = EQW(object_spectra, continuum, lower_wav, upper_wav, do_errs)
        #ew = numpy.squeeze(ew)
        ew, lower_wav, upper_wav, err_ew = find_EW(object_spectra, continuum, line_looked_for, lower_wav, upper_wav, do_errs)
        #ew, lower_wav, upper_wav, err_ew = find_EW_withsplotfunc(object_spectra, continuum, line_looked_for, lower_wav, upper_wav, do_errs)
    else:
        #ew, lower_wav, upper_wav = EQW(object_spectra, continuum, lower_wav, upper_wav)        
        # determine equivalent width by finding the max or the min of the line
        ew, lower_wav, upper_wav = find_EW(object_spectra, continuum, line_looked_for, lower_wav, upper_wav)
        #ew, lower_wav, upper_wav = find_EW_withsplotfunc(object_spectra, continuum, line_looked_for, lower_wav, upper_wav)
        #print 'I USED THE function that U want'
    ew = float(ew)
    F = ew * C #* (-1)   # with the actual equivalent width definition
    #print (lower_wav+upper_wav)/2.0, ' F=', F, 'ew=', ew, '  from ', lower_wav, '  to ', upper_wav
    if do_errs != None:
        if lower_wav < 2000.:
            n = 0 #this gets the percentage error of the continuum
        elif (lower_wav >= 2000) and (lower_wav < 5000.):
            n = int(len(do_errs[1])/2)
        elif lower_wav >= 5000.:
            n = -1
        err_perc = do_errs[1][n]/continuum[1][n]  #this gets the percentage error of the continuum
        errC = C * err_perc
        err_F =  numpy.sqrt( F**2*( (err_ew/ew)**2 + (errC/C)**2 - 2*((err_ew*errC)**2/(ew*C)) ))
        #print 'F, err_F, C, errC, err_perc', F, err_F, C, errC, err_perc
        return F, C, err_F, ew, lower_wav, upper_wav, err_ew
    else:
        return F, C, ew, lower_wav, upper_wav

def theo_cont(wave_arr, scale_factor=1.0):
    '''
    Since these are theoretical data, the continuum is by definition at 1 when normalizing
    the data array has to be the wavelength element, the result is going to be an array
    of both wavelength and flux.
    '''
    cont_temp = []
    for _ in wave_arr:
        cont_temp.append(1.0 * scale_factor)
    theoretical_cont = numpy.array([wave_arr, cont_temp]) 
    return theoretical_cont


#### READING SPECIFIC LINES
def readlines_EWabsRelHbeta():
    '''This function specifically reads the lines of the file to correct for underlying absorption: abs_EWs_rel_Hbeta.txt'''
    # Hydrogen
    H_lines = []
    H_EWratios = []
    # Helium
    He_lines = []
    He_EWratios = []
    # List of the data contained in the file assuminf path /Users/home_directory/Documents/AptanaStudio3/science/science/spectrum/abs_EWs_rel_Hbeta.txt
    text_file = os.path.abspath('../../science/science/spectrum/abs_EWs_rel_Hbeta.txt')
    f = open(text_file, 'r')
    list_rows_of_file = f.readlines()
    f.close()
    for row in list_rows_of_file:
        # Disregard comment symbol
        if '#' not in row:
            # Split each row into columns
            line_data = row.split()
            wav = float(line_data[1])
            ewabs = float(line_data[2])
            if 'HI' in row:
                # append the element into each column in the Hydrogen cols_in_file
                H_lines.append(wav)
                H_EWratios.append(ewabs)
            if 'HeI' in row:
                # append the element into each column in the Hydrogen cols_in_file
                He_lines.append(wav)
                He_EWratios.append(ewabs)
    Hline_and_EWs = [H_lines, H_EWratios]
    Heline_and_EWs = [He_lines, He_EWratios]
    return Hline_and_EWs, Heline_and_EWs

def readreddCorr(obs_wav_arr):
    '''This function reads specifically the file with Seaton's law 1979.'''
    obs_wav_arr = numpy.array(obs_wav_arr)
    # List of the data contained in the file assuming path /Users/home_directory/Documents/AptanaStudio3/science/science/spectrum/reddeningCorSeaton79.txt
    text_file = os.path.abspath('../../science/science/spectrum/reddeningCorSeaton79.txt')
    wavs, f_lambda = numpy.loadtxt(text_file, skiprows=2, usecols=(0,1), unpack=True)
    f_lambda_obs = []
    for w in obs_wav_arr:
        if w in wavs:
            f = f_lambda[wavs == w]
            f = float(f)
        else:
            f = numpy.interp(w, wavs, f_lambda)
        f_lambda_obs.append(f)
    return f_lambda_obs


#### EQUIVALENT WIDTH FUNCTIONS 
def EQW_line_fixed(line_arr, line_cont, line, width=10.0):
    '''
    This function determines the EW integrating over a fixed interval.
    *** THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    # line is the rest wavelength of the line of interest
    # line_arr is the tuple array of wavelength and flux for the line
    # line_cont is also a tuple of wavelength and flux (=1 if it is theoretical)
    # Limits to find the EWQ
    '''
    lo = line - (width/2.0)
    up = line + (width/2.0)
    lolim, _ = find_nearest(line_arr[0,:], lo)
    uplim, _ = find_nearest(line_arr[0,:], up)    
    new_line_arr_x, new_line_arr_y = selection(line_arr[0,:], line_arr[1,:], lolim, uplim)
    _, new_cont_arr_y = selection(line_cont[0,:], line_cont[1,:], lolim, uplim)    
    '''
    x, y = line_arr
    x_cont, y_cont = line_cont
    new_line_arr_x = x[(lo >= x) & (up <= x)]
    new_line_arr_y = y[(lo >= x) & (up <= x)]
    new_cont_arr_y = y_cont[(lo >= x_cont) & (up <= x_cont)]
    '''
    # In case arrays do not have the exact same wavelengths, I am removing the second to last element
    # so that the width remains the same.
    if len(new_line_arr_y) > len(new_cont_arr_y):
        new_line_arr_x = numpy.delete(new_line_arr_x, new_line_arr_x[len(new_line_arr_x)-2])
        new_line_arr_y = numpy.delete(new_line_arr_y, new_line_arr_y[len(new_line_arr_y)-2])
    elif len(new_line_arr_y) < len(new_cont_arr_y):
        new_cont_arr_y = numpy.delete(new_cont_arr_y, new_cont_arr_y[len(new_cont_arr_y)-2])
    # Finding the average step for the integral
    # automatic adjustable lambda
    N = len(new_line_arr_x)
    print('Closest points in array to lower limit and upper limit: %f, %f' % (new_line_arr_x[0], new_line_arr_x[N-1]))    
    print('Fixed width = %f,   Actual width = %f' % (width, new_line_arr_x[N-1]-new_line_arr_x[0]))
    i = 0
    diffs_list = []
    for j in range(1, N):
        point_difference = new_line_arr_x[j] - new_line_arr_x[i]
        diffs_list.append(point_difference)
        i = i + 1
    diffs_arr = numpy.array(diffs_list)
    dlambda = numpy.sum(diffs_arr) / numpy.float(N)    
    # fixed lambda
    #dlambda = 0.1  # in Angstroms
    # Actually solving the eqw integral
    # my method
    difference = 1-(new_line_arr_y / new_cont_arr_y)
    eqw = sum(difference) * dlambda * (-1)   # the -1 is because of the definition of EQW
    return (eqw)

def half_EQW_times2(data_arr, cont_arr, line, wave_limit, right_side=True):
    '''
    This function determines half of the EW from the chosen side and then it multiplies it by 2.
    *** THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    # data_arr = data array that contains both wavelength and flux
    # cont_arr = continuum array that also contains both wavelength and flux
    # ARRAYS MUST BE SAME DIMENSIONS
    # line = where the integration should begin
    # wave_limit = final point of the integration
    '''
    if right_side == True:
        lower = line
        upper = wave_limit
    else:
        lower = wave_limit
        upper = line
    half_eqw, _, _ = EQW(data_arr, cont_arr, lower, upper)
    eqw = half_eqw * 2.0
    return(eqw)

def fill_EWarr(data_arr, cont_arr, lolim, uplim, elements):
    width = uplim - lolim
    wavelength_list = []
    flux_list = []
    w_cont_list = [] 
    flux_cont_list = []
    increment = width / float(elements)
    #print 'width, increment', width, increment
    wavelength_list.append(lolim)
    w_cont_list.append(lolim)
    flolim = numpy.interp(lolim, data_arr[0], data_arr[1])
    #print 'INTERP  lolim, flolim', lolim, flolim
    flux_list.append(flolim)
    flolim_cont = numpy.interp(lolim, cont_arr[0], cont_arr[1])
    flux_cont_list.append(flolim_cont)
    w = lolim
    for _ in range(0, elements-1):
        w = w + increment
        f = numpy.interp(w, data_arr[0], data_arr[1])
        wavelength_list.append(w)
        #print 'INTERP  w, f', w, f
        flux_list.append(f)
        w_cont_list.append(w)
        fc = numpy.interp(w, cont_arr[0], cont_arr[1])
        flux_cont_list.append(fc)
    wavelength = numpy.array(wavelength_list)
    flux = numpy.array(flux_list)
    w_cont = numpy.array(w_cont_list)
    flux_cont = numpy.array(flux_cont_list)
    return wavelength, flux, w_cont, flux_cont

def EQW(data_arr, cont_arr, lower, upper, do_errs=None):
    '''
    This function detemrines the equivalent width integrating over the interval given by the lower and upper limits.
    *** THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    # data_arr = data array that contains both wavelength and flux
    # cont_arr = continuum array that also contains both wavelength and flux
    # ARRAYS MUST BE SAME DIMENSIONS
    # lower = where the integration should begin
    # upper = final point of the integration
    # THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    '''
    # Finding closest wavelength to the desired lower and upper limits
    lower, _ = find_nearest(data_arr[0], lower)
    upper, _ = find_nearest(data_arr[0], upper)
    #print('Closest points in array to lower limit and upper limit: %f, %f' % (lower, upper))
    #width = upper - lower
    #print('Actual width = %f' % (width))
    # Finding the line arrays to use in the integration
    wavelength, flux = selection(data_arr[0], data_arr[1], lower, upper)
    _, flux_cont = selection(cont_arr[0], cont_arr[1], lower, upper)
    
    # Interpolate so that the flux selection array has 10 elements
    #elements = 100
    #wavelength, flux, _, flux_cont = fill_EWarr(data_arr, cont_arr, lower, upper, elements)
    # Finding the average step for the integral
    N = len(wavelength)
    i = 0
    diffs_list = []
    for j in range(1, N):
        point_difference = wavelength[j] - wavelength[i]
        diffs_list.append(point_difference)
        i = i + 1
    dlambda = sum(diffs_list) / float(N)
    # Actually solving the eqw integral
    difference = 1 - (flux / flux_cont)
    eqw = sum(difference) * dlambda * (-1)   # the -1 is because of the definition of EQW        
    if do_errs != None:
        errs_fluxes, errs_continuum = do_errs        
        err_a = []
        err_b = []
        for f, c, ef, ec in zip(flux, flux_cont, errs_fluxes, errs_continuum):
            # Let  a = flux / flux_cont, then error in a is
            ea = f/c * numpy.sqrt((ec/c)**2 + (ef/f)**2)
            err_a.append(ea)
            # Let  b = 1-a, then the error**2 in b is
            eb = ea*ea
            err_b.append(eb)
        tot_err_diff = numpy.sqrt(sum(err_b))
        err_ew = dlambda * numpy.sqrt(tot_err_diff)
        #print 'lower_limit =', lower, '  upper_limit =', upper, '  ew =', eqw, '+-', err_ew
        return (eqw, lower, upper, err_ew)
    else:
        #final_width = upper - lower
        #print('center=', (upper+lower)/2.0,'  final_width = %f' % final_width, '    ew=', eqw)
        #print 'lower_limit =', lower, '  upper_limit =', upper,'  ew =', eqw
        return (eqw, lower, upper)

def EQW_iter(data_arr, cont_arr, line, guessed_width=3.0):
    '''
    This function tries to determine automatically the width of the line, starting with an
    educated guess.
    # line = line rest wavelength of the line of interest
    # data_arr = data array that contains both wavelength and flux
    # cont_arr = continuum array that also contains both wavelength and flux
    # guessed_width = minimum width of the line
    
    ***** data_arr  and  cont_arr  HAVE to have the same dimensions
    
    # THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    '''
    lower = line - (guessed_width/2)  # lower = where the integration should begin
    upper = line + (guessed_width/2)  # upper = final point of the integration
    eqw_central, lolim, uplim = EQW(data_arr, cont_arr, lower, upper)
    #print('eqw_central, lolim, uplim', eqw_central, lolim, uplim)
    #tolerance = (eqw_initial) ** 0.5
    tolerance = numpy.fabs(eqw_central / 1000.0)  # according to the decimal places wanted in precission
    #print('tolerance = %f' % (tolerance))
    # first increase of EQW
    increment1 = 1.0
    eqw_increase1, lo_increase1, up_increase1 = EQW(data_arr, cont_arr, lolim-increment1, uplim+increment1) 
    # second increase of EQW
    increment2 = increment1 * 2.0
    eqw_increase2, _, _ = EQW(data_arr, cont_arr, lolim-increment2, uplim+increment2) 
    #print('eqw_increase = %f, lo_increase = %f, up_increase = %f' % (eqw_increase, lo_increase, up_increase))
    difference1 = numpy.fabs(numpy.fabs(eqw_central) - numpy.fabs(eqw_increase1))
    difference2 = numpy.fabs(numpy.fabs(eqw_central) - numpy.fabs(eqw_increase2))
    
    if (difference1 <= tolerance) and (difference2 <= (tolerance*2.0)):
        eqw = eqw_increase1
        lolim = lo_increase1
        uplim = up_increase1
        return (eqw, lolim, uplim)
    else:
        single_increment = 1.0
        # first increase of EQW
        increment1 = increment1 + single_increment
        eqw_increase1, lo_increase1, up_increase1 = EQW(data_arr, cont_arr, lolim-increment1, uplim+increment1) 
        # second increase of EQW
        increment2 = increment1 * 2.0
        eqw_increase2, _, _ = EQW(data_arr, cont_arr, lolim-increment2, uplim+increment2) 
        difference1 = numpy.fabs(numpy.fabs(eqw_central) - numpy.fabs(eqw_increase1))
        difference2 = numpy.fabs(numpy.fabs(eqw_central) - numpy.fabs(eqw_increase2))
        # error check
        if (lo_increase1 == data_arr[0, 0]) or (up_increase1 == data_arr[0, len(data_arr[0])-1]):
            print('OOPS! Reached limit of array. Could not find an EQW!')
            eqw = '--'            
            lolim = '--'
            uplim = '--'
            return(eqw, lolim, uplim)
    eqw = eqw_increase1
    lolim = lo_increase1
    uplim = up_increase1
    #final_width = uplim - lolim
    #print('final_width = %f' % final_width)
    return (eqw, lolim, uplim)

def recenter(line_wave, line_flux, original_width):
    sq_fluxes = []
    for lf in line_flux:
        sq_fluxes.append(lf*lf)
    line_peak = max(sq_fluxes)
    idx_line_peak = sq_fluxes.index(line_peak)
    recentered = line_wave[idx_line_peak]
    lower = recentered - (original_width/2.0)
    upper = recentered + (original_width/2.0)
    return recentered, lower, upper

def determine_if_is_peak(line_wave, line_flux, original_center):
    _, originalcenter_idx =  find_nearest(line_wave, original_center)
    originalcenter_flx = line_flux[originalcenter_idx]
    _, left_originalcenter_idx =  find_nearest(line_wave, original_center-1.5)
    left_originalcenter_flx = line_flux[left_originalcenter_idx]
    _, right_originalcenter_idx =  find_nearest(line_wave, original_center+1.5)
    right_originalcenter_flx = line_flux[right_originalcenter_idx]
    # Determine if the line is a max or a min
    originalcenter_is_max = False
    if originalcenter_flx > 0.0:
        originalcenter_is_max = True
    originalcenter_is_peak = False
    left_is_lessthan_original_center = False
    right_is_lessthan_original_center = False
    if originalcenter_is_max:                               # we have an emission line
        if originalcenter_flx > left_originalcenter_flx:
            left_is_lessthan_original_center = True
        if originalcenter_flx > right_originalcenter_flx:
            right_is_lessthan_original_center = True
    if originalcenter_is_max == False:                      # we have an absorption line
        if originalcenter_flx < left_originalcenter_flx:
            left_is_lessthan_original_center = True
        if originalcenter_flx < right_originalcenter_flx:
            right_is_lessthan_original_center = True
    if left_is_lessthan_original_center and right_is_lessthan_original_center:
        originalcenter_is_peak = True
    return originalcenter_is_peak

def find_first2peaks(line_wave, line_flux, original_width):
    line_wave_copy = copy.deepcopy(line_wave)
    line_flux_copy = copy.deepcopy(line_flux)
    peak1, _, _ = recenter(line_wave_copy, line_flux_copy, original_width)
    peak1_is_peak = determine_if_is_peak(line_wave, line_flux, peak1)
    _, idx = find_nearest(line_wave_copy, peak1)
    line_flux_copy[idx] = 0.0
    peak2, _, _ = recenter(line_wave_copy, line_flux_copy, original_width)
    return peak1, peak2   # these are the 2 WAVELENGTHS ath which there is a peak flux

def find_splotEW(line_wave, line_flux, flux_cont, do_errs):
    '''This function is the translation into python of the cl scripts: sumflux.x and eqwidth.x'''
    sum1 = 0.0
    rsum = 0.0
    esum = 0.0
    csum = 0.0
    sum2 = 0.0
    slope = (flux_cont[-1] - flux_cont[0]) / (line_wave[-1] - line_wave[0])
    scale = max(line_flux)
    if scale <= 0.0:
        scale = 1.0
    fc_first = flux_cont[0]
    for w, f, fc in zip(line_wave, line_flux, flux_cont):
        rampval = fc_first + slope * (w - line_wave[0])
        sum1 = sum1 + fc
        rsum = rsum + rampval
        esum = esum + (1.0 - f/rampval)
        delta = (f - rampval)/scale
        csum = csum + numpy.abs(delta)**1.5 * w
        sum2 = sum2 + numpy.abs(delta)**1.5
    sum1 = sum1 + line_flux[0] + line_flux[-1]
    rsum = rsum + flux_cont[0] + flux_cont[-1]
    esum = esum + (1.0 - line_flux[0] / flux_cont[0]) + (1.0 - line_flux[-1] / flux_cont[-1])
    delta = (line_flux[-1] - flux_cont[-1]) / scale
    csum = csum + numpy.abs(delta)**1.5 * line_wave[-1]
    sum2 = sum2 + numpy.abs(delta)**1.5
    if sum2 != 0.0:
        ctr = csum / sum2
    else:
        ctr = 0.0
    # Correct for angstroms/channel
    wpc = numpy.abs((line_wave[-1] - line_wave[0]) / float(len(line_wave) - 1))
    sum1 = sum1 * wpc
    rsum = rsum * wpc
    esum = esum * wpc * -1 # this is to follow the convention of possitive=emission, negative=absorption
    # Compute difference in flux between ramp and spectrum
    flux_diff = sum1 - rsum
    # compute eq width of feature using ramp midpoint as continuum
    cont = 0.5 * (flux_cont[0] + flux_cont[-1])
    # Print on status line
    print 'center = %0.3f,   eqw = %0.4f,   continuum = %0.6e,   flux = %0.6e' % (ctr, esum, cont, flux_diff)
    if do_errs != None:
        return esum, line_wave[0], line_wave[-1], 1.0
    else:
        return esum, line_wave[0], line_wave[-1]

def find_EW_withsplotfunc(data_arr, cont_arr, nearest2line, low, upp, do_errs=None):
    '''
    This function recenters the line according to the max or min (emission or absorption) and then adjusts 
    according to the min difference between the flux and the continuum.
    nearest2line = the target line that we want to measure
    low = closest point in the wavelength array to lower part of the predefined width of the line
    upp = closest point in the wavelength array to upper part of the predefined width of the line
    '''
    original_width = float(upp) - float(low)
    lower = float(low)
    upper = float(upp) + 3.5   # due to redshift
    elements = 100
    line_wave, line_flux, _, _ = fill_EWarr(data_arr, cont_arr, lower, upper, elements)
    _, lower, upper = recenter(line_wave, line_flux, original_width)
    line_wave, line_flux, _, flux_cont = fill_EWarr(data_arr, cont_arr, lower, upper, elements)
    if do_errs != None:
        eqw, lolim, uplim , err_ew = find_splotEW(line_wave, line_flux, flux_cont, do_errs)
        return (eqw, lolim, uplim, err_ew)
    else:
        eqw, lolim, uplim = find_splotEW(line_wave, line_flux, flux_cont, do_errs)
        return (eqw, lolim, uplim)
    
def find_EW(data_arr, cont_arr, line_looked_for, low, upp, do_errs=None):
    '''
    This function recenters the line according to the max or min (emission or absorption) and then adjusts 
    according to the min difference between the flux and the continuum.
    line_looked_for = the target line that we want to measure
    low = closest point in the wavelength array to lower part of the predefined width of the line
    upp = closest point in the wavelength array to upper part of the predefined width of the line
    '''
    lower = float(low)
    upper = float(upp) + 3.5   # due to redshift
    original_width = float(upp) - float(low)
    original_center = (float(upp) + float(low))/2.0
    #print 'ORIGINAL CENTER=', original_center
    #print 'ORIGINALS:  lower =', lower, ' upper =', upper, '   width =', original_width
    # Recenter the line according to the max in the sqared fluxes and determine if we have an emission or 
    # absorption at the closest point to the target line
    elements = 10
    line_wave, line_flux, _, flux_cont = fill_EWarr(data_arr, cont_arr, lower, upper, elements)
    line_is_emission = False
    nearest2target_line, _ = find_nearest(line_wave, line_looked_for)
    print 'Nearest wavelegth in line array to the target line =', nearest2target_line
    recenter0, _, _ = recenter(line_wave, line_flux, original_width)
    recenter0_flx = numpy.interp(recenter0, line_wave, line_flux)
    recenter0_cont = numpy.interp(recenter0, line_wave, flux_cont)
    if recenter0_flx >= recenter0_cont:
        line_is_emission = True
        print' line is EMISSION'
    else:
        print' line is ABSORPTION'
    #if line_looked_for == 4640.0:
    #    line_is_emission = True
    #elif line_looked_for == 4650.0:
    #    line_is_emission = True
    # WAIT! Before recentering to the max/min in the line array, make sure that the original center is a not peak or closer to it,
    # this part is just in case there is a close peak that is higher/lower than the original center.
    # Frirst normalize the spectrum
    norm_flx = []
    for lw, lf, cf in zip(line_wave, line_flux, flux_cont):
        nf = numpy.abs(lf) / numpy.abs(cf)
        if lf < 0.0:
            nf = nf * -1
        norm_flx.append(nf)
        if lf >= 5230.0:
            print 'normalization at ', lw, nf, '  Flux=', lf, '  Continuum=', cf, raw_input()
    # According to the appropriate type of line, find the peak
    positive_fluxes = []
    negative_fluxes = []
    for f in norm_flx:
        if f >= 1.0:
            positive_fluxes.append(f)
        else:
            negative_fluxes.append(f)
    print 'lengths of positive_fluxes and negative_fluxes', len(positive_fluxes), len(negative_fluxes)
    if len(positive_fluxes) == 0.0:
        line_is_emission = False
    if len(negative_fluxes) == 0.0:
        line_is_emission = True
    if line_is_emission:
        peak_flx = max(positive_fluxes)
    else:
        peak_flx = min(negative_fluxes)
    idx_peak = norm_flx.index(peak_flx)
    recenter1 = line_wave[idx_peak]
    print '   1st recenter could be at', recenter1
    # Now, determine recenter2: this is the overall max/min, regardless if it is absorption or emission
    # Just in case the continuum went negative, make it positive! 
    # This works for absorption and emission....  :)    
    sq_fluxes = []
    for lf in line_flux:
        sq_fluxes.append(lf*lf)
    line_peak = max(sq_fluxes)
    idx_line_peak = sq_fluxes.index(line_peak)
    recenter2 = line_wave[idx_line_peak]
    print '   2nd recenter: ', recenter2
    # Determine which recenter is closer to the original center
    diff_recenter1 = numpy.abs( recenter1 - original_center )
    diff_recenter2 = numpy.abs( recenter2 - original_center )
    if diff_recenter1 < diff_recenter2:       #  1st recenter closer to original center
        peak_wav = recenter1
    elif diff_recenter1 >= diff_recenter2:     #  2nd recenter closer to original center
        peak_wav = recenter2
    lower = peak_wav - (original_width/2.0)
    upper = peak_wav + (original_width/2.0)
    # with the new lower and upper points, redetermine a line array
    line_wave, line_flux, _, flux_cont = fill_EWarr(data_arr, cont_arr, lower, upper, elements)
    # Find out in the line array where is the point closest to the continuum to recenter again according to it
    norm_flx = []
    for lw, lf, cf in zip(line_wave, line_flux, flux_cont):
        nf = numpy.abs(lf) / numpy.abs(cf)
        norm_flx.append(nf)
        #print 'normalization at ', lw, nf, '  Flux=', lf, '  Continuum=', cf
    min_diff = min(norm_flx, key=lambda x:abs(x-1.0))
    min_idx = norm_flx.index(min_diff)
    wav_min_diff = line_wave[min_idx] 
    print 'the continuum crossing point is at:', wav_min_diff
    if wav_min_diff < recenter1:
        lolim = wav_min_diff
        uplim = wav_min_diff + original_width
    else:
        uplim = wav_min_diff
        lolim = wav_min_diff - original_width
    new_center = (lolim + uplim) / 2.0
    print 'ORIGINAL CENTER=', original_center, '   NEW CENTER OF THE LINE=', new_center
    print 'limits:   ', lolim, uplim
    # Determine the equivalent width
    if do_errs != None:
        #eqw, lolim, uplim , err_ew = EQW(data_arr, cont_arr, lolim, uplim, do_errs)
        #print 'lower_limit =', lolim, '  upper_limit =', uplim, '  ew =', eqw, '+-', err_ew
        eqw, lolim, uplim , err_ew = find_splotEW(line_wave, line_flux, flux_cont, do_errs)
        return (eqw, lolim, uplim, err_ew)
    else:
        #eqw, lolim, uplim = EQW(data_arr, cont_arr, lolim, uplim)
        #print('center=', new_center,'  final_width = %f' % uplim - lolim, '    ew=', eqw)
        #print 'lower_limit =', lolim, '  upper_limit =', uplim, '  ew =', eqw
        eqw, lolim, uplim = find_splotEW(line_wave, line_flux, flux_cont, do_errs)
        return (eqw, lolim, uplim)
    
def fill_arr2limit(x, y, ylimit, fromX, toX, resolution_of_spectra):
    '''This function interpolates in order to slice and fill-in the numpy array of x and y down to the desired limit.
    # ylimit is an array with the same the units of the y array, i.e. the continuum array
    # fromX toX are the limiting values to fill the slice of x array
    # x and y must have the same dimensions
    # resolution_of_spectra is the number subtracted/added to fromX/toX in order to "give room" to slice the array.'''
    fromX_y = numpy.interp(fromX, x, y)      # Find the value in y at the lower x limit
    fromX_c = numpy.interp(fromX, x, ylimit) # Find the value in the continuum at the lower x limit
    toX_y = numpy.interp(toX, x, y)          # Find the value in y at the upper x limit
    toX_c = numpy.interp(toX, x, ylimit)     # Find the value in the continuum at the upper x limit
    newx_list = []    # for the wavelengths
    newy_list = []    # for the fluxes
    newc_list = []    # for the continua
    # add the left-most values of x and y to the lists
    newx_list.append(fromX)
    newy_list.append(fromX_y)
    newc_list.append(fromX_c)
    # slice the original array so that the function is faster
    sliceX = x[(x >= fromX-resolution_of_spectra) & (x <= toX+resolution_of_spectra)]
    sliceY = y[(x >= fromX-resolution_of_spectra) & (x <= toX+resolution_of_spectra)]
    sliceC = ylimit[(x >= fromX-resolution_of_spectra) & (x <= toX+resolution_of_spectra)]
    # add the values in the data between the x limits
    for xi, yi, ci in zip(sliceX, sliceY, sliceC):
        if xi > fromX:
            newx_list.append(xi)
            newy_list.append(yi)
            newc_list.append(ci)
    # Now if the y values are less than the ylimit, set those equal to the ylimit 
    for i in range(len(newy_list)):
        if newy_list[i] < newc_list[i]:
            newy_list[i] = newc_list[i]
    # Find the average value of the continuum in order to find where in x, y crosses c
    avgC = sum(newc_list)/len(newc_list)
    x_at_avgC = numpy.interp(avgC, sliceY, sliceX)
    c_at_avgC = numpy.interp(x_at_avgC, sliceX, sliceC) # find the true value of the continuum at that wavelength
    # find the index in which to insert the crossing point
    for i in range(len(newx_list)):
        if newx_list[i] > x_at_avgC:
            idx_limit = i
            break
    newx_list.insert(idx_limit, x_at_avgC)
    newy_list.insert(idx_limit, c_at_avgC)
    newc_list.insert(idx_limit, c_at_avgC)
    # add the interpolated desired end
    for i in range(len(newx_list)):
        #print 'newx_list[i], x_at_limit', newx_list[i], x_at_limit
        if newx_list[i] > toX:
            idx_end = i
            break
    newx_list.insert(idx_end, toX)
    newy_list.insert(idx_end, toX_y)
    newc_list.insert(idx_limit, toX_c)

    # For best fitting purposes, we will end the line where there is a min in the flux towards that wavelength
    maxpt = max(newy_list)      
    maxpt_idx = newy_list.index(maxpt)
    sliceylist = newy_list[maxpt_idx:idx_end]
    minpt2end = min(sliceylist)
    minpt2end_idx = newy_list.index(minpt2end)
    fromX_idx = newx_list.index(fromX)
    slicexlist = newx_list[fromX_idx:minpt2end_idx+1]
    sliceylist = newy_list[fromX_idx:minpt2end_idx+1]
    sliceclist = newc_list[fromX_idx:minpt2end_idx+1]
    # turn the lists into numpy array
    newx = numpy.array(slicexlist) 
    newy = numpy.array(sliceylist) 
    newc = numpy.array(sliceclist)
    # interpolate a couple of points, one of each side of the maximum to "help" the fit
    left_max_x = (newx_list[maxpt_idx] + newx_list[maxpt_idx-1]) / 2.0
    left_max_y = numpy.interp(left_max_x, newx, newy)
    left_max_c = numpy.interp(left_max_x, newx, newc)
    right_max_x = (newx_list[maxpt_idx] + newx_list[maxpt_idx+1]) / 2.0
    right_max_y = numpy.interp(right_max_x, newx, newy)
    right_max_c = numpy.interp(right_max_x, newx, newc)
    #print 'max=', newx_list[maxpt_idx], '  inserting before', left_max_x, ' after', right_max_x
    max_idx = numpy.where(newx == newx_list[maxpt_idx])
    #print 'max_idx', max_idx[0][0], newx[max_idx[0][0]]
    newx = numpy.insert(newx, max_idx[0][0], left_max_x)
    newx = numpy.insert(newx, max_idx[0][0]+2, right_max_x)
    newy = numpy.insert(newy, max_idx[0][0], left_max_y)
    newy = numpy.insert(newy, max_idx[0][0]+2, right_max_y)
    newc = numpy.insert(newc, max_idx[0][0], left_max_c)
    newc = numpy.insert(newc, max_idx[0][0]+2, right_max_c)
    # Now make sure these lists go down all the way to the ylimit or continuum fromX toX
    newx = numpy.append(newx, toX)
    newy = numpy.append(newy, toX_c)
    newc = numpy.append(newc, toX_c)
    toXplus1 = newx_list[idx_end+1]
    newx = numpy.append(newx, toXplus1)
    newy = numpy.append(newy, toX_c)
    newc = numpy.append(newc, toX_c)
    #print 'fromX=', fromX, fromX_y, fromX_c
    #print 'toX=', toX, toX_y, toX_c
    #print 'inserts:', newx_list[fromX_idx]
    #print newx
    #print newy
    #print newc
    '''
    # turn the lists into numpy array
    newx = numpy.array(newx_list) 
    newy = numpy.array(newy_list) 
    newc = numpy.array(newc_list)
    print 'shape of line + resolution', numpy.shape(newx)
    newx = newx[(newx >= fromX) & (newx <= toX)]
    newy = newy[(newx >= fromX) & (newx <= toX)]
    newc = newc[(newx >= fromX) & (newx <= toX)]
    print 'shape of array of line only', numpy.shape(newx)
    '''
    return newx, newy, newc
    
def deblend_line(object_spectra, contum_spectra, catal_wavelengths, obs_wavelengths, tot_flx, tot_cont, lines2deblend, width_of_lines, plot_fit=False):
    '''This function deblends the input line. IT WAS PROGRAMED FOR EMISSION LINES! 
    lines2deblen = list of the central wavelengths that compose the blend
    width_of_lines = width of the componing lines (is the same for all the components of the blend)
    if plot_fit = True, the function shows plot of deblend.'''
    midwav = sum(lines2deblend) / float(len(lines2deblend))
    print 'lines2deblend:', lines2deblend
    print 'midwav = ', midwav
    width = width_of_lines + 6.0    # this adds 3 A from each side to "give room" for the fit 
    lolim = midwav - width/2.0
    uplim = midwav + width/2.0
    resolution_of_spectra = 3.0
    inix, iniy, _ = fill_arr2limit(object_spectra[0], object_spectra[1], contum_spectra[1], lolim, uplim, resolution_of_spectra)
    # Determine the arrays for the lines , as well as the total flux, continuum, equivalent width
    gaussfits = []
    acoeff = []
    sigmas = []
    xs = []
    ys = []
    diff = 0.0
    norm_constant =  1.0e-15
    for line in lines2deblend:
        print 'line and width:', line, width_of_lines
        if line in catal_wavelengths:
            obsline_idx = catal_wavelengths.index(line)
            obsline = obs_wavelengths[obsline_idx]
            lolim = obsline - width/2.0
            uplim = obsline + width/2.0
            print 'center, lolim, uplim:', obsline, lolim, uplim
            x, y, _ = fill_arr2limit(object_spectra[0], object_spectra[1], contum_spectra[1], lolim, uplim, resolution_of_spectra)
            fromX = obsline - width_of_lines/2.0
            toX = obsline + width_of_lines/2.0
            newx = x[(x >= fromX) & (x <= toX)]
            newy = y[(x >= fromX) & (x <= toX)]
            xs.append(newx)
            ys.append(newy)
            newy = newy / norm_constant # Normalizing the flux to avoid computing problems
            a = 1.0                     # first guess
            p0 = [a, line, 1.5]         # p0 is the initial guess for the fitting coefficients (a, mean, and sigma)
            coeff, _ = optimize.curve_fit(gaus_function, newx, newy, p0)
            print 'coeffs: a, mean, sigma =', coeff
            acoeff.append(coeff[0])
            sigmas.append(coeff[-1]) 
            p = [coeff[0], coeff[1]+diff, coeff[-1]]
            diff = lines2deblend[1] - lines2deblend[0]           
            gf = gaus_function(x, *p) * 1.0e-15
            gaussfits.append(gf)
            if plot_fit:
                pyplot.plot(x, gf,'b:',label='initial guess')
    # Sum the initial gaussians
    p0 = [acoeff[0], acoeff[1], lines2deblend[0], lines2deblend[1], sigmas[0], sigmas[1]]
    iniy = iniy / norm_constant
    coeffini, _ = optimize.curve_fit(gauss_sum, inix, iniy, p0=p0)
    y_initsum = gauss_sum(inix, coeffini[0], coeffini[1], lines2deblend[0], lines2deblend[1], sigmas[0], sigmas[1]) * norm_constant
    #y_initsum = gaus_function(inix, acoeff[0], lines2deblend[0], sigmas[0])*norm_constant + gaus_function(inix, acoeff[1], lines2deblend[1], sigmas[1])*norm_constant
    print 'coeffini, y_initsum', coeffini[0], coeffini[1], y_initsum
    pyplot.plot(inix, y_initsum,'b-.',label='initial guess')
    initial_param4residuals = numpy.array(p0)
    plsq = optimize.leastsq(residuals, p0[:], args=(inix, iniy))
    print plsq
    # Determine the estimate
    y1 = gaus_function(xs[0], plsq[0][0], plsq[0][2], plsq[0][4]) * norm_constant
    y2 = gaus_function(xs[1], plsq[0][1], plsq[0][3], plsq[0][5]) * norm_constant
    y_est = gaus_function(inix, plsq[0][0], plsq[0][2], plsq[0][4])*norm_constant + gaus_function(inix, plsq[0][1], plsq[0][3], plsq[0][5])*norm_constant
    # Return values to original state
    iniy = iniy * norm_constant
    if plot_fit:
        pyplot.plot(object_spectra[0], object_spectra[1],'k:',label='data')
        pyplot.plot(contum_spectra[0], contum_spectra[1],'r:',label='continuum')
        pyplot.plot(inix, iniy,'k')
        pyplot.plot(xs[0], y1,'c--',label='line1')
        pyplot.plot(xs[1], y2,'m--',label='line2')
        pyplot.plot(inix, y_est,'g--',label='sum')
        pyplot.xlim(lolim-25.0, uplim+25.0)
        pyplot.ylim(0.0, 2.5e-15)
        #pyplot.ylim(tot_flx-(tot_flx*0.3), tot_flx+(tot_flx*0.1))
        pyplot.legend()
        pyplot.title('Line deblend with Gaussian Fit')
        pyplot.xlabel('Wavelength  [$\AA$]')
        pyplot.ylabel('Flux  [ergs/s/cm$^2$/$\AA$]')
        pyplot.show()
    #return 

#### Gaussian fit
def gaus_function(x, norm, meanX, sig):
    '''This function finds the gaussian y corresponding point.
    x = array of x-values
    RETURNS: array of y-values '''
    a = norm/(sig*numpy.sqrt(2*numpy.pi))
    return a * numpy.exp(-(x-meanX)**2 / (2.0*sig**2))

def gauss_sum(x, norm1, norm2, mean1, mean2, sig1, sig2):
    '''This function finds the sum of 2 gaussians.
    x = array of x-values
    RETURNS: y-point in gaussian '''
    norm = norm1 * norm2
    a = norm/numpy.sqrt(2.0*numpy.pi*(sig1**2 + sig2**2))    
    xtot = x + x
    return a * numpy.exp(-(xtot-(mean1+mean2))**2 / (2.0*(sig1**2 + sig2**2)))

def gaus_fit(x, y):
    sig, mean = find_std(x)
    # p0 is the initial guess for the fitting coefficients (mean and sigma)
    norm = 0.5 # initial guess
    p0 = [norm, mean, sig]
    coeff, _ = optimize.curve_fit(gaus_function, x, y, p0=p0)
    gf = gaus_function(x, *coeff)
    #print 'these are the coefficients:  norm=', coeff[0], '  mean=', coeff[1], '  sigma=', coeff[2]
    return gf, sig

def residuals(p, y, x):
    n1, n2, m1, m2, sd1, sd2 = p
    y_fit = gaus_function(x, n1, m1, sd1) + gaus_function(x, n2, m2, sd2)
    err = y - y_fit
    return err
  
def deblend_with_gauss(object_spectra, contum_spectra, catal_wavelengths, obs_wavelengths, tot_flx, tot_cont, lines2deblend, width_of_lines, plot_fit=False):
    # to "simplify" the data, its zero will be the continuum and it will not consider adyacent lines
    width = width_of_lines + 5.0    # this adds 2.5 A from each side to "give room" for the fit 
    for line in lines2deblend:
        print 'line and width:', line, width
        if line in catal_wavelengths:
            obsline_idx = catal_wavelengths.index(line)
            obsline = obs_wavelengths[obsline_idx]
            lolim = obsline - width/2.0
            uplim = obsline + width/2.0
            print 'center, lolim, uplim:', obsline, lolim, uplim
            nearest2lolim, _ = find_nearest(object_spectra[0], lolim)
            nearest2uplim, _ = find_nearest(object_spectra[0], uplim)
            x = object_spectra[0][(object_spectra[0] >= nearest2lolim) & (object_spectra[0] <= nearest2uplim)]
            y = object_spectra[1][(object_spectra[0] >= nearest2lolim) & (object_spectra[0] <= nearest2uplim)]
            line_cont = contum_spectra[1][(contum_spectra[0] >= nearest2lolim) & (contum_spectra[0] <= nearest2uplim)]
    # Remove the "irrelevant" information
    for i in range(len(y)):
        y[i] = y[i] - line_cont[i]
    # Initial conditions
    #mainline = (lines2deblend[0] + lines2deblend[1])/2.0
    mean1, mean2, sig1, sig2 = [lines2deblend[0], lines2deblend[1], 1.0, 1.0] 
    param = [mean1, mean2, sig1, sig2]
    y_init = gaus_function(x, mean1, sig1) + gaus_function(x, mean2, sig2)
    p = [lines2deblend[0], lines2deblend[1], sig1, sig1]
    plsq = optimize.leastsq(residuals, param, args=(y, x))
    # Determine the estimate
    y_est = gaus_function(x, plsq[0][0], plsq[0][2]) + gaus_function(x, plsq[0][0] + plsq[0][1], plsq[0][3])
    for i in range(len(y_est)):
        y_est[i] = y_est[i] + line_cont[i]
    if plot_fit:
        pyplot.plot(object_spectra[0], object_spectra[1],'k',label='data')
        pyplot.plot(contum_spectra[0], contum_spectra[1],'r:',label='continuum')
        pyplot.plot(x, y_init, 'g--', label='Starting Guess')
        pyplot.plot(x, y_est, 'b--', label='Fitted')
        pyplot.xlim(lolim-50.0, uplim+50.0)
        pyplot.ylim((tot_flx*0.1)-tot_flx, tot_flx+(tot_flx*0.1))
        pyplot.legend()
        pyplot.title('Line deblend with Gaussian Fit')
        pyplot.xlabel('Wavelength  [$\AA$]')
        pyplot.ylabel('Flux  [ergs/s/cm$^2$/$\AA$]')
        pyplot.show()
    
#### Full width half maximum 
def FWHM(sig):
    # This can only be runned after the gaus_fit function
    fwhm = 2 * numpy.sqrt(2 * numpy.log(2)) * sig
    print 'FWHM = ', fwhm
    return fwhm

#### Error functions
def find_std(arr):
    '''
    This function determines the standard deviation of the given array.
    '''
    N = float(len(arr))
    mean = numpy.sum(arr) / N
    diff2meansq_list = []
    for a in arr:
        diff = a - mean
        diffsq = diff * diff
        diff2meansq_list.append(diffsq)
    std = ( 1.0/(N-1.0) * sum(diff2meansq_list) )**(0.5)
    #print 'sigma = ', std
    return std, mean

def absolute_err(measurement, true_value):
    '''This function finds the absolute value of the error given the measurement and true values.'''
    abs_err = numpy.fabs(numpy.fabs(measurement) - numpy.fabs(true_value))
    return abs_err

def relative_err(measurement, true_value):
    '''This function finds the fractional error given the measurement and true values. It returns a percentage.'''
    rel_err = ( absolute_err(measurement, true_value) / numpy.fabs(true_value) ) * 100
    return rel_err



############################################################################################
# REBINNING FUNCTIONS AND RESOLVING POWER
def rebin(arr, factor):
    ''' arr: array-like tuple
    factor: rebin_factor tuple
    *** A cubic spline function is used for performing the interpolation in the zoom function'''
    return ndimage.interpolation.zoom(arr, factor, order=1)

def correct_rebin(arr1, arr2):
    ''' Sometimes there is a less line in the resulting rebinned arrays. This function corrects for that.
    - Both arrays have to have the ONE dimension
    - Deleting the second element in each array (not the first so that the width remains the same)
    - The axis in which delete is 0, meaning the entire row.
    - THIS FUNCTION RETURNS:
                            2 arrays of same shape '''
    N_arr1 = len(arr1[0,:])
    N_arr2 = len(arr2[0,:])
    if N_arr1 > N_arr2:
        new_point_x = arr2[0,N_arr2-2] + 0.5
        new_point_y = numpy.interp(new_point_x, arr2[0,:], arr2[1,:])
        arr2_x = numpy.insert(arr2[0,:], N_arr2-2, new_point_x)
        arr2_y = numpy.insert(arr2[1,:], N_arr2-2, new_point_y)
        arr2 = numpy.array([arr2_x, arr2_y])
    elif N_arr1 < N_arr2:
        new_point_x = arr1[0,N_arr1-2]+0.5
        new_point_y = numpy.interp(new_point_x, arr1[0,:], arr1[1,:])
        arr1_x = numpy.insert(arr1[0,:], N_arr1-2, new_point_x)
        arr1_y = numpy.insert(arr1[1,:], N_arr1-2, new_point_y)
        arr1 = numpy.array([arr1_x, arr1_y])
    return(arr1, arr2)

def resolving_power(line, arr):
    '''This function determines the resolving power R at the line wavelength.
    # arr must be a numpy array of wavelength and flux. '''
    closest_line, idxline = find_nearest(arr[0,:], line)
    new_delta_lambda = closest_line - arr[0, idxline-1]
    Resolution = int(line / new_delta_lambda)
    return (Resolution)

def rebin_to_desired_rows(arr1, arr2, desired_rows):
    '''This function simply finds the rebinned new arrays with the zoom function.
    both arrays have to be in form of wavelength and flux numpy arrays.
    THIS FUNCTION RETURNS:
        - rebinned arr1
        - rebinned arr2
        - rebinning factor of arr1
        - rebinning factor of arr2
    '''
    arr1_shape = numpy.shape(arr1)
    arr2_shape = numpy.shape(arr2)
    #print('Shapes of arrays: lines = %s  and  continuum = %s' % (repr(arr1_shape), repr(arr2_shape)))
    if arr1_shape[1] > arr2_shape[1]:
        big_arr = arr1
        big_arr_shape = arr1_shape
        small_arr = arr2
        small_arr_shape = arr2_shape
    elif arr2_shape[1] > arr1_shape[1]:
        big_arr = arr1
        big_arr_shape = arr1_shape
        small_arr = arr2
        small_arr_shape = arr2_shape
    elif arr2_shape[1] == arr1_shape[1]:
        if arr1_shape[1] > desired_rows:
            new_factor = float(desired_rows) / float(arr1_shape[1]) 
        elif arr1_shape[1] < desired_rows:
            new_factor = float(desired_rows) / float(arr1_shape[1])
        elif arr1_shape[1] == desired_rows:
            new_factor = 1.0
        arr1_rebinned = rebin(arr1, (1, new_factor))
        arr2_rebinned = rebin(arr2, (1, new_factor))
        return (arr1_rebinned, arr2_rebinned, new_factor, new_factor)
 
    if big_arr_shape[1] > desired_rows:
        big_arr_factor = float(desired_rows) / float(big_arr_shape[1])
    elif big_arr_shape[1] < desired_rows:
        big_arr_factor = float(big_arr_shape[1]) / float(desired_rows)
    elif big_arr_shape[1] == desired_rows:
        big_arr_factor = 1.0
    
    if small_arr_shape[1] > desired_rows:
        small_arr_factor = float(desired_rows) / float(small_arr_shape[1])
    elif small_arr_shape[1] < desired_rows:
        small_arr_factor = float(desired_rows) / float(small_arr_shape[1])
    elif small_arr_factor[1] == desired_rows:
        small_arr_factor = 1.0
    
    #print('big_arr_factor = %f   ---   small_arr_factor = %f' % (big_arr_factor, small_arr_factor))
    big_arr_rebinned = rebin(big_arr, (1, big_arr_factor))
    small_arr_rebinned = rebin(small_arr, (1, small_arr_factor))
    shape_big_arr = numpy.shape(big_arr_rebinned)
    shape_small_arr = numpy.shape(small_arr_rebinned)
    
    # Fix the lengths to make sure they are equal
    if shape_big_arr != shape_small_arr:
        big_arr_rebinned, small_arr_rebinned = correct_rebin(big_arr_rebinned, small_arr_rebinned)
    return (big_arr_rebinned, small_arr_rebinned, big_arr_factor, small_arr_factor)
    
def rebin_arrays_for_desired_resolution(desired_delta_lambda, line, lines_arr, cont_arr, guessed_rows=500):
    '''In other words this function smoothens the spectra to get the desired desired_delta_lambda.
    Arrays must be numpy arrays.
    guessed_rows to speed up code in the iterations.'''
    R_initial = resolving_power(line, lines_arr)
    delta_lambda_initial = line / float(R_initial)
    print('Initial Resolving_power of lines array = %i  ----  Initial delta_lambda = %f' % (R_initial, delta_lambda_initial))
    delta_lambda = delta_lambda_initial
    desired_rows = guessed_rows
    #print('Guessed rows = %s' % repr(guessed_rows))
    lines_rebin, cont_rebin, new_lines_factor, new_cont_factor = rebin_to_desired_rows(lines_arr, cont_arr, desired_rows)
    new_R_initial = resolving_power(line, lines_rebin)
    new_delta_lambda_initial = line / float(new_R_initial)
    #print('NEW Resolving_power = %i  ----  NEW delta_lambda = %f' % (new_R_initial, new_delta_lambda_initial))
    if new_delta_lambda_initial > desired_delta_lambda:
        #print 'new_delta_lambda_initial > desired_delta_lambda'
        desired_rows = guessed_rows + 200
        lines_rebin, cont_rebin, new_lines_factor, new_cont_factor = rebin_to_desired_rows(lines_arr, cont_arr, desired_rows)
        new_R_initial = resolving_power(line, lines_rebin)
        new_delta_lambda_initial = line / float(new_R_initial)
        #print('NEW Resolving_power = %i  ----  NEW delta_lambda = %f' % (new_R_initial, new_delta_lambda_initial))

    while (numpy.fabs(desired_delta_lambda - delta_lambda) > 0.0035):
        lines_rebin, cont_rebin, new_lines_factor, new_cont_factor = rebin_to_desired_rows(lines_arr, cont_arr, desired_rows)
        #print('New shape of rebinned arrays: lines, continuum', lines_rebin.shape, cont_rebin.shape)
        R = resolving_power(line, lines_rebin)
        delta_lambda = line / float(R)
        desired_rows = desired_rows - 1
        #print('Initial Resolving_power = %i  ----  Initial delta_lambda = %f' % (R_initial, delta_lambda_initial))
        #print('**** NEW R and delta_lambda:   %f -- %f' % (R, delta_lambda))
        #print 'Decreasing rows! Started at: %i, now at: %i' % (guessed_rows, desired_rows)
    smoothing_factor = float(R_initial) / float(R)
    #if R_initial >= R:
    #    smoothing_factor = float(R_initial) / float(R)
    #elif R_initial < R:
    #    smoothing_factor = float(R) / float(R_initial)
    print 'Factor of increase or decrease for: lines_array = %f and continuum_array = %f' % (new_lines_factor, new_cont_factor)
    print 'Delta_lambda = %f,  Resolving Power = %i,  smoothing_R_factor = %f' % (delta_lambda, R, smoothing_factor)
    print 'Decreased rows from an initial guess of %i to %i' % (guessed_rows, desired_rows)
    return (lines_rebin, cont_rebin)

def rebin_one_arr_to_desired_rows(arr, desired_rows):
    _, rows = arr.shape
    if rows > desired_rows:    
        factor = float(desired_rows) / float(rows)
    elif rows < desired_rows:     
        factor = float(desired_rows) / float(rows)
    elif rows == desired_rows:
        factor = 1.0
    rebinned_arr = rebin(arr, (1, factor))
    return (rebinned_arr, factor)
    
def rebin_one_arr_to_desired_resolution(desired_delta_lambda, line, arr, guessed_rows=500):
    '''
    This function does the same as "rebin_arrays_for_desired_resolution" but only for one array.
    RETURNS:  - rebinned array
              - smoothing/increasing factor with respect to initial Resolution Power
    '''
    R_initial = resolving_power(line, arr)
    delta_lambda_initial = line / float(R_initial)
    print('Initial Resolving_power of lines array = %i  ----  Initial delta_lambda = %f' % (R_initial, delta_lambda_initial))
    delta_lambda = delta_lambda_initial
    desired_rows = guessed_rows
    while (abs(desired_delta_lambda - delta_lambda) > 0.0015):
        rebinned_arr, _ = rebin_one_arr_to_desired_rows(arr, desired_rows) 
        R = resolving_power(line, rebinned_arr)
        delta_lambda = line / float(R)
        desired_rows = desired_rows - 1
    smoothing_R_factor = float(R_initial) / float(R)
    return (rebinned_arr, smoothing_R_factor)

def find_dispersion(wavs):
    original_rows = float(len(wavs))
    i = 0
    disp_list = []
    for j in range(1, len(wavs)):
        disp = wavs[j] - wavs[i]
        disp_list.append(disp)
        i = i + 1
    original_disp = sum(disp_list) / original_rows
    return original_disp

def rebin_spec2disp(desired_dispersion, arr):
    ''' 
    This function uses the desired dispersion and the initial wavelength to rebin the spectra.
        Array must be numpy arrays: wavelengths and fluxes.
    RETURNS:  - rebinned array
    '''
    wavs, flxs = arr
    original_disp = find_dispersion(wavs)
    print 'Original dispersion is:  %f' % original_disp
    '''
    resPow_avg = []
    for line in wavs:
        resPow = resolving_power(line, arr)
        resPow_avg.append(resPow)
    avgR = int(sum(resPow_avg)) / len(wavs)
    print 'avgR', avgR
    '''
    if original_disp == desired_dispersion:
        print 'The desired dispersion is the spectrum dispersion.' 
        return arr
    elif original_disp != desired_dispersion:
        print 'Calculating new dispersion...'
        rebinned_wavs = []
        rebinned_flxs = []
        rebinned_wavs.append(wavs[0])
        rebinned_flxs.append(flxs[0])
        reb_w = wavs[0] + desired_dispersion
        reb_f = numpy.interp(reb_w, wavs, flxs)
        rebinned_wavs.append(reb_w)
        rebinned_flxs.append(reb_f)
        end_while = False
        end_wave = rebinned_wavs[-1]
        #print 'w, f', reb_w, reb_f, '      end_wave', end_wave
        while end_while == False:
            reb_w = reb_w + desired_dispersion
            reb_f = numpy.interp(reb_w, wavs, flxs)
            rebinned_wavs.append(reb_w)
            rebinned_flxs.append(reb_f)            
            end_wave = rebinned_wavs[-1]
            #print 'w, f', reb_w, reb_f, '      end_wave', end_wave
            if (end_wave > wavs[-1]-5.0) and (end_wave < wavs[-1]+5.0):
                end_while = True
    rebinned_disp = find_dispersion(rebinned_wavs)
    print 'rebinned_disp = %0.2f' % rebinned_disp
    return numpy.array([rebinned_wavs, rebinned_flxs])
    '''
    ### Text file of Resolving Power of every wavelength
    resol = open(path_results+'CMFGENResolvingPower.txt', 'w+')
    resPow_avg = []
    for line in A:
        resPow = spectrum.resolving_power(line, data_lines)
        resPow_avg.append(resPow)
        print >> resol, line, resPow 
        print('wavelength and R', line, resPow)
    avgR = int(sum(resPow_avg)) / len(A)
    print >> resol, 'Average Reolving Power of CMFGEN lines file = '
    print >> resol, avgR
    resol.close()
    print('Average Reolving Power of CMFGEN lines file = %i' % (avgR))
    if line == None:
        for line in arr[0]:
            rebinned_arr, _ = rebin_one_arr_to_desired_resolution(desired_delta_lambda, line, arr, guessed_rows)
    '''

def rebin2AperPix(original, desired, wavsflx_arr):
    new =  original / desired
    print 'rebinning factor', new
    rebin_factor = (1, new)
    rebinned_arr = rebin(wavsflx_arr, rebin_factor)
    return rebinned_arr

def do_rebin(spectrum_arr, continuum_arr, desired_rows=500):
    '''### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRA ARRAY HAVE SAME DIMENSIONS'''
    orig_factors = (1, 1) # Numbers that will mumtiply colums and rows
    continuum_factor, spec_factor = factor_newshape(continuum_arr, spectrum_arr, orig_factors[1], desired_rows)
    rebin_spec = rebin(spectrum_arr, (1, spec_factor))    
    rebin_cont = rebin(continuum_arr, (1, continuum_factor))
    # Since arrays have same dimensions, continuum_factor = spec_factor
    return rebin_spec, rebin_cont, spec_factor    

def get_factors_and_rebin(spectrum_arr, continuum_arr, desired_rows=600):
    '''
    THIS FUNCTIONS DOES EVERYTHING AT ONCE FOR SPECTRA WITH SPECTRA AND CONT ARRAYS OF DIFFERENT DIMENSIONS
    # spectrum_arr = numpy array of wavelength and flux
    # continuum_arr = numpy array of wavelength and flux
    # desired_rows = number of lines for output file, by default it is 600
    THIS FUNCTION RETURNS: 
    #     rebinned_line_array, 
    #     rebinned_continuum_array, 
    #     and the factors by which each one has been decreased/increased: 
    #         new_continuum_factor, 
    #         new_lineintensity_factor
    '''
    # the next two functions return a tuple for columns and rows
    spec_factors = get_factor_lineintensity(continuum_arr, spectrum_arr) 
    cont_factors = get_factor_continuum(continuum_arr, spectrum_arr)
    shape_line = spectrum_arr.shape
    shape_cont = continuum_arr.shape
    if shape_line[1] > shape_cont[1]:
        factors = spec_factors
    elif shape_line[1] < shape_cont[1]:
        factors = cont_factors
    elif shape_line[1] == shape_cont[1]:
        factors = spec_factors
    # using those factors so that the number of rows is the same 
    ### USING factor_newshape function
    new_continuum_factor, new_lineintensity_factor = factor_newshape(continuum_arr, spectrum_arr, factors[1], desired_rows)    
    rebin_line = rebin(spectrum_arr, (1, new_lineintensity_factor))    
    rebin_cont = rebin(continuum_arr, (1, new_continuum_factor))
    return rebin_line, rebin_cont, new_continuum_factor, new_lineintensity_factor

def factor_newshape(continuum_arr, lineintensity_arr, lineintensity_factor, desired_rows=500):
    percent = 1.00
    oldshape = lineintensity_arr.shape
    newshape = lineintensity_arr.shape
    new_lineintensity_factor = lineintensity_factor * percent
    continuum_arr_shape = continuum_arr.shape
    while percent > 0.00:
        if newshape[1] == desired_rows:
            break
        new_lineintensity_factor = lineintensity_factor * percent
        temp_lineintensity_arr = rebin(lineintensity_arr, (1, new_lineintensity_factor))
        newshape = temp_lineintensity_arr.shape
        percent -= 0.0001
        continuum_arr_shape = continuum_arr.shape
    new_continuum_factor = (oldshape[1] * new_lineintensity_factor) / continuum_arr_shape[1]
    #print("Percent: %f  Newshape: %s  New Line Intensity Factor: %f" % (percent, newshape, new_lineintensity_factor))
    #print(newshape)
    return(new_continuum_factor, new_lineintensity_factor)

def same_rebin_factor(arr1, arr2):
    ''' This is the factor that rebins _fin and _cont by the same amount
    ### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRUM ARRAYS ARE NOT SAME DIMENSIONS
    '''
    temp_factor_same = 1.0
    arr1_shape = numpy.shape(arr1)
    arr2_shape = numpy.shape(arr2)
    if arr1_shape[1] > arr2_shape[1]:
        big_arr = arr1
        small_arr = arr2
    elif arr2_shape[1] > arr1_shape[1]:
        big_arr = arr2
        small_arr = arr1
    elif arr1_shape[1] == arr2_shape[1]:
        factor_same = temp_factor_same
        #print 'Arrays are the same dimensions. Factor is 1.0'
        return factor_same
    small_arr_columns, small_arr_rows = numpy.shape(small_arr)
    big_arr_columns, big_arr_rows = numpy.shape(big_arr)
    # The number of colums in the small array has to be equal to the columns in line big array
    if big_arr_columns != small_arr_columns:
        raise ValueError('Number of columns in files do not match')
    #print big_arr_rows, small_arr_rows
    factor_same = (big_arr_rows/small_arr_rows)**0.5 
    return factor_same

def get_factor_lineintensity(continuum_arr, lineintensity_arr):
    ''' ### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRUM ARRAYS ARE NOT SAME DIMENSIONS'''
    factor_same = same_rebin_factor(continuum_arr, lineintensity_arr)
    _, continuum_rows = continuum_arr.shape
    lineintensity_columns, lineintensity_rows = lineintensity_arr.shape
    #print continuum_arr.shape, lineintensity_arr.shape
    factor_lineintensity = (continuum_rows * factor_same) / lineintensity_rows
    return (lineintensity_columns, factor_lineintensity)

def get_factor_continuum(continuum_arr, lineintensity_arr):
    '''### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRUM ARRAYS ARE NOT SAME DIMENSIONS'''
    factor_continuum = same_rebin_factor(continuum_arr, lineintensity_arr)
    continuum_columns, _ = continuum_arr.shape
    #print('This is the factor_eq for cont', factor_continuum)
    return (continuum_columns, factor_continuum)

# MY ATTEMPT TO CREATE REBINNING FUNCTIONS WITHOUT USING ZOOM=SPLINE3
def insert_point_left(arr, reference_wavelength):
    ''' this is a two dimensional numpy array. '''
    px1, idx_px1 = find_nearest(arr[0,:], reference_wavelength)
    idx_px2 = idx_px1 - 1
    px2 = arr[0,idx_px2]
    px3 = midpoint(px1, px2)
    py3 = numpy.interp(px3, arr[0,:], arr[1,:])
    new_arrX = numpy.insert(arr[0,:], idx_px2, px3)
    new_arrY = numpy.insert(arr[1,:], idx_px2, py3)
    new_arr = numpy.array([new_arrX, new_arrY])
    new_reference_wavelength = arr[0, idx_px2]
    return (new_arr, new_reference_wavelength)

def insert_point_right(arr, reference_wavelength):
    ''' this is a two dimensional numpy array. '''
    px1, idx_px1 = find_nearest(arr[0,:], reference_wavelength)
    idx_px2 = idx_px1 + 1
    px2 = arr[0,idx_px2]
    px3 = midpoint(px1, px2)
    py3 = numpy.interp(px3, arr[0,:], arr[1,:])
    new_arrX = numpy.insert(arr[0,:], idx_px2, px3)
    new_arrY = numpy.insert(arr[1,:], idx_px2, py3)
    new_arr = numpy.array([new_arrX, new_arrY])
    new_reference_wavelength = arr[0, idx_px2]
    return (new_arr, new_reference_wavelength)

def just_rebin_interpol(arr, factor, reference_wavelength):
    '''
    arr is a numpy array of wavelength and flux.
    This function returns a numpy array of the same shape as arr.
    '''
    _, rows =  numpy.shape(arr)
    desired_arr_rows = rows * factor
    new_arr, new_reference_wavelength = insert_point_left(arr, reference_wavelength)
    _, new_arr_rows = numpy.shape(new_arr)
    count_points = 1
    while new_arr_rows != desired_arr_rows:
        if count_points%2==0:
            new_arr, new_reference_wavelength = insert_point_right(new_arr, new_reference_wavelength)
            _, new_arr_rows = numpy.shape(new_arr)
            count_points = count_points + 1
        else:
            new_arr, new_reference_wavelength = insert_point_left(new_arr, new_reference_wavelength)
            _, new_arr_rows = numpy.shape(new_arr)
            count_points = count_points + 1
        #print ('desired number of rows: ', desired_arr_rows)
        #print ('new shape so far is: ', numpy.shape(new_arr))
    return (new_arr)

def smoothTriangle(data, degree, dropVals=False):
    """performs moving triangle smoothing with a variable degree."""
    """note that if dropVals is False, output length will be identical
    to input length, but with copies of data at the flanking regions"""
    triangle = numpy.array(range(degree) + [degree] + range(degree)[::-1]) + 1
    smoothed = []
    for i in range(degree, len(data) - degree * 2):
        point = data[i:i + len(triangle)] * triangle
        smoothed.append(sum(point) / sum(triangle))
    if dropVals: return smoothed
    smoothed = [smoothed[0]] * (degree + degree / 2) + smoothed
    while len(smoothed) < len(data):smoothed.append(smoothed[-1])
    return smoothed

def rebinning_interpol(arr1, arr2, reference_wavelength):
    '''
    - Both arrays are two dimensional arrays: Shapes should be the same, length can be different.
    - The reference_wavelength is from where to start adding points to the small array.
    - Use this function when arrays of lines and continuum are not the same length.
    - This function interpolates the smaller data set to make it the same as the big one.
    #### THIS FUNCTION RETURNS: the rebinned array and the corresponding factor.
    '''
    _, rows1 = numpy.shape(arr1)
    _, rows2 = numpy.shape(arr2)
    rows_arr1 = float(rows1)
    rows_arr2 = float(rows2)
    if rows_arr1 < rows_arr2:
        arr_small = arr1
        arr_big = arr2
        factor = (rows_arr2 ) / rows_arr1
    elif rows_arr1 > rows_arr2:
        arr_small = arr2
        arr_big = arr1
        factor = (rows_arr1 ) / rows_arr2
    elif rows_arr1 == rows_arr2:
        print 'No need to use this function. Use the "just_rebin_interpol" function with the same factor for both arrays.'
        exit()
    print ('Shape of big array: ', numpy.shape(arr_big))
    print ('Shape of small array: ', numpy.shape(arr_small))    
    print('Interpolating...')
    rebinned_arr_small = just_rebin_interpol(arr_small, factor, reference_wavelength)
    print ('Shape of new "small" array: ', numpy.shape(rebinned_arr_small))    
    return (rebinned_arr_small, factor)

############################################################################################
# OTHER USEFUL FUNCTIONS
def write_1d(filename, data):
    '''filename: file name
    data: tuple of values'''
    try:
        print("Writing 1d file: %s" % (filename))
        output = numpy.column_stack((data))
        numpy.savetxt(filename, output, delimiter=' ')
    except IOError as e:
        print("%s: %s" % (filename, e.strerror))
        return False
    return True

def find_nearest(arr, value):
    '''
    This function gives the content and the index in the array of the number that is closest to 
    the value given.
    '''
    idx=(numpy.abs(arr-value)).argmin()
    return arr[idx], idx

def find_nearest_within(arr, value, threshold):
    '''
    This function gives the content in the array of the number that is closest to 
    the value given, within threshold away from value.
    '''
    half_thres = threshold / 2.0
    choped_arr = arr[(arr >= value-half_thres) & (arr <= value+half_thres)]
    if len(choped_arr) == 0:
        return 0.0
    diff = numpy.abs(choped_arr - value)
    diff_min = min(diff)
    return numpy.squeeze(choped_arr[diff==diff_min])

def closest(thelist, value) :
    '''
    This function gives the content in the list of the number that is closest to 
    the value given.
    '''
    return min((abs(value - i), i) for i in thelist)[1]
    
def find_index_in_list(the_list, item):
    for idx in range(0, len(the_list)):
        if the_list[idx] == item:
            return(idx)
        
def midpoint(p1, p2):
    return p1 + (p2-p1)/2

def findXinY (Xarr, Yarr, value):
    '''This function finds value in array Y and returns the corresponding value in array X
    # Wanted value is in array Xarr
    # Given value is in array Yarr
    '''
    x = 0.0
    for i in range(0, len(Xarr)):
        if Yarr[i] == value:
            x = Xarr[i] 
    if x == 0.0:
        raise Exception("Value %f not found in Yarr" % (value))
        exit(1)
    return x

def convert2abs(thelist):
    for i in range(0, len(thelist)):
        thelist[i] = abs(thelist[i])
    return thelist

def selection(arr_x, arr_y, lower, upper):
    # These arrays must be one dimensional numpy arrays and they both must have same length
    arr_x_selected = arr_x[(upper >= arr_x) & (lower <= arr_x)]
    arr_y_selected = arr_y[(upper >= arr_x) & (lower <= arr_x)]
    return (arr_x_selected, arr_y_selected)

def extrapolate_arr2value(wav_and_flux_arr, wav, left=True):
    '''
    This function extrapolates the array from left/right to value.
    # wav_and_flux_arr = 2D array of wavelengths and fluxes
    # value = the wavelength we want to extraplotate to
    # left = True  will make the extrapolation go towards lower wavelengths
           = False  will extrapolate to higher wavelengths
    Function returns extrapolated 2D array. 
    '''
    last_flux_point = numpy.interp(wav, wav_and_flux_arr[0], wav_and_flux_arr[1])
    diff = numpy.fabs(numpy.fabs(wav) - numpy.fabs(wav_and_flux_arr[0][0]))
    new_wavs = []
    new_flux = []
    increment = 2.0
    if left == True:
        inter_wav = wav_and_flux_arr[0][0] - increment
    elif left == False:
        inter_wav = wav_and_flux_arr[0][len(wav_and_flux_arr[0])-1] + increment 
    while diff > 0.5:
        #print 'inter_wav', inter_wav      
        if left == True:
            inter_wav = inter_wav - increment
            if inter_wav < wav:
                inter_wav = wav 
                new_flux_point = numpy.interp(inter_wav, wav_and_flux_arr[0], wav_and_flux_arr[1])
                break
        if left == False:
            inter_wav = inter_wav + increment
            if inter_wav > wav: 
                inter_wav = wav 
                new_flux_point = numpy.interp(inter_wav, wav_and_flux_arr[0], wav_and_flux_arr[1])
                break            
        new_flux_point = numpy.interp(inter_wav, wav_and_flux_arr[0], wav_and_flux_arr[1])
        new_flux.append(new_flux_point)
        new_wavs.append(inter_wav)
        diff = numpy.fabs(numpy.fabs(wav) - numpy.fabs(inter_wav))
    new_flux.append(last_flux_point)
    new_wavs.append(wav)
    #print 'wav, last_wav', wav, inter_wav
    new_arr = numpy.array([new_wavs, new_flux])
    return new_arr
    
def combine(a):
    '''Combine tuple into a numpy.array'''
    return numpy.array(a)

def div(tuple1, tuple2):
    cont = combine(tuple1)
    fin = combine(tuple2)
    """
    #cont_rf = rebin_factor(cont, fin) # returns factor tuple
    #fin_rf = rebin_factor(cont, fin, factor_only=False) # returns scaled tuple
    cont_rf = factor_newshape(cont, fin, ) # returns factor tuple
    fin_rf = factor_newshape(cont, fin, factor_only=False) # returns scaled tuple    
    rebin_cont = rebin(cont, cont_rf)
    rebin_fin = rebin(fin, fin_rf)
    """
    A = fin[0]
    CGS = fin[1] / cont[1]
    return numpy.array((A, CGS))

############################################################################################
# THIS CLASS CONVERTS FROM JANSKIES AND HERTZ TO CGS UNITS 
class Spectrum:
    def __init__(self, filename):
        numpy.set_printoptions(precision=20, suppress=False, threshold='nan')
        self.filename = filename
        self.packed = None
        self.hz = None
        self.jy = None
        self.err = None
        self.A = None
        self.cgs = None
        self.A_selected = None
        self.cgs_selected = None
        
        self.loaded = False
        self.load_packed_1d(filename)
        self.hz = self.packed[0]
        self.jy = self.packed[1]
        self.err = self.packed[2]
        if len(self.hz) != len(self.jy):
            print "Dimensions do not match"     
        self.A, self.cgs = self.convert_inline()
    
    def isloaded(self):
        return self.loaded
    
    def load_packed_1d(self, filename):
        self.packed = numpy.loadtxt(
                                    filename, 
                                    dtype=numpy.float64,
                                    delimiter=' ', 
                                    usecols=(0,1,2),
                                    unpack=True
                                    )
        if self.packed != None:
            self.loaded = True
            return True
        
    def convert_inline(self):
        print("Converting values:")
        print("hztoA..."),
        A = self.hztoA(self.hz)
        print("done")
        print("jytocgs..."),
        cgs = self.jytocgs(self.hz, self.jy)
        print("done")
        return A, cgs
        
    ''' hztoA mockup '''
    def hztoA(self, w):
        Alst = []
        for i in range(0, len(w)):
                A = 2.998e18 / (w[i]*1e15) #+ 3
                Alst.append(A)
        A = numpy.array(Alst)
        return A
    
    ''' jytocgs mockup '''
    def jytocgs(self, w, f):
        cgslst = []
        for i in range(0, len(f)):
                cgs = f[i] * 2.998e-05 * ((w[i] * 1e15) / 2.998e18) ** 2
                cgslst.append(cgs)
        cgs = numpy.array(cgslst)
        return cgs

    def save(self):
        f = self.filename
        tosave = [
                  [f+'_packed', (self.packed)],
                  [f+'_converted', (self.A, self.cgs, self.err)],
                 ]
        if self.A_selected != None and self.cgs_selected != None:
            tosave.append([f+'_selected', (self.A_selected, self.cgs_selected, numpy.zeros(self.cgs_selected.shape))])
            
        for filename, tdata in tosave:
            write_1d(filename, (tdata))
    
    def save2(self, f):
        tosave = [
                  [f, (self.A, self.cgs, self.err)],
                 ]
        if self.A_selected != None and self.cgs_selected != None:
            tosave.append([f+'_selected', (self.A_selected, self.cgs_selected, numpy.zeros(self.cgs_selected.shape))])
            
        for filename, tdata in tosave:
            write_1d(filename, (tdata))   
    def select_set(self, lower, upper):
        xlist = []
        ylist = []
        for i in range(0, len(self.hz)):
            if self.A[i] >= lower and self.A[i] <= upper:
                xlist.append(self.A[i])
                ylist.append(self.cgs[i])
        
        self.A_selected = numpy.array(xlist)
        self.cgs_selected = numpy.array(ylist)
    
    def select_jy_from_hz(self, lower, upper):
        rlist = []
        for i in range(0, len(self.hz)):
            if self.hz[i] >= lower and self.hz[i] <= upper:
                rlist.append(self.jy[i])
        r = numpy.array(rlist)
        return r

    def select_cgs_from_A(self, lower, upper):
        xlist = []
        rlist = []
        for i in range(0, len(self.A)):
            if self.A[i] >= lower and self.A[i] <= upper:
                xlist.append(self.A[i])
                rlist.append(self.cgs[i])
        r = numpy.array([xlist, rlist])
        return r
    
    def select_hz_from_jy(self, lower, upper):
        rlist = []
        for i in range(0, len(self.jy)):
            if self.jy[i] >= lower and self.jy[i] <= upper:
                rlist.append(self.hz[i])
        r = numpy.array(rlist)
        return r
        
    def select_A_from_cgs(self, lower, upper):
        rlist = []
        for i in range(0, len(self.cgs)):
            if self.cgs[i] >= lower and self.cgs[i] <= upper:
                rlist.append(self.A[i])
        r = numpy.array(rlist)
        return r        
    
    def show_chart(self):
        ''' Print ranged values '''
        w = self.A
        f = self.cgs
        wavelength = self.hz
        flux = self.jy
        logging.basicConfig(filename=os.path.join('/tmp', 'chart.log'),level=logging.DEBUG)
        for i in range(0, len(w)):
            logging.info('orig_w(%.20f) :: A_w(%.20f) => (orig_f(%.20f) :: cgs_f(%.20f)' % 
                    (wavelength[i], w[i], flux[i], f[i]) )
            pprint( 'orig_w(%.20f) :: A_w(%.20f) => (orig_f(%.20f) :: cgs_f(%.20f)' % 
                    (wavelength[i], w[i], flux[i], f[i]) )