import numpy
import math
import os
import logging
from scipy import ndimage
from pprint import pprint
from scipy import stats
from matplotlib import pyplot
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
    std = find_std(flux_arr)
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
    # List all the standard deviations
    std_list = []
    std_list.append(std)
    # Add the clipped fluxes
    clipd_fluxes_list = []
    for f in flx_clip:
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
        print 'Window: ', window_lo, window_up
        f_win = object_spec[1][(object_spec[0] > window_lo) & (object_spec[0] <= window_up)]
        std, flx_clip = iterate_sigma_clipdflx_usingMode(f_win, sigmas_away)
        print 'sigma = ', std
        for f in flx_clip:
            clipd_fluxes_list.append(f)
        std_list.append(std)
    #print 'std_list', std_list
    clipd_arr = numpy.array([object_spec[0], clipd_fluxes_list])
    return clipd_arr, std_list

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

def fit_continuum(object_name, object_spectra, z, order=5, sigmas_away=3.0, window=300, plot=True, z_correct=True, normalize=True):
    '''
    This function shifts the object's data to the rest frame (z=0). The function then fits a 
    continuum to the entire spectrum, omitting the lines windows (it interpolates 
    in that region). It then CAN normalizes the entire spectrum.
    The lines it looks for are those in the lines2fit.txt file.
    REQUIREMENTS:
    # object_spectra must be a 2D numpy array of wavelengths and fluxes
    # z is expected to be the redshift of the object
    # nth is the order of the polynomial, default is 5
    # thresold_fraction = freaction (percentage) to multiply the threshold. This modifies the width
                          of the flux band in which to allow interpolation of fluxes
    # window_width = the size of the spectrum window to be analyzed.
        * The default window size: 150 A but it can be set to take into account the whole spectrum.
    FUNCTION RETURNS:
    # 2D numpy array of redshift-corrected wavenegths and fluxes.
    # 2D continuum numpy array of wavenegths and fluxes.
    '''
    print 'Calculating continuum...'
    # Bring the object to rest wavelenght frame using 1+z = lambda_obs/lambda_theo - 1
    if z_correct == True:
        print '    *** Wavelengths corrected for redshift.'
        w_corr = object_spectra[0] / (1+float(z))
    else:
        print '    *** Wavelengths NOT YET redshifted...'
        w_corr = object_spectra[0]
    #DIVIDE THE SPECTRUM INTO SMALLER WINDOWS TO FIND THE LOCAL MODE
    # this is the array to find the continuum with
    corr_wf = numpy.array([w_corr, object_spectra[1]])
    wf, std_list = get_spec_sigclip(corr_wf, window, sigmas_away)
    # Polynolial of the form y = Ax^5 + Bx^4 + Cx^3 + Dx^2 + Ex + F
    coefficients = numpy.polyfit(wf[0], wf[1], order)
    polynomial = numpy.poly1d(coefficients)
    f_pol = polynomial(wf[0])
    fitted_continuum = numpy.array([wf[0], f_pol])
    chi2_list = test_min_chi2_of_polynomial(fitted_continuum, std_list, window)
    avg_chi2 = sum(chi2_list) / float(len(chi2_list))
    #print 'avg_chi2 for this polynomial: ', avg_chi2
    '''
    #print 'object_spectra[1][0], wf[1][0], fitted_continuum[1][0]', object_spectra[1][0], wf[1][0], fitted_continuum[1][0]
    ###### USE THIS PART WHEN WANTING TO COMPARE WITH THE FLUX-MODE CLIPPING
    ### Alternatively, use the flux mode to clip
    mode_wf = clip_flux_using_modes(corr_wf, window, threshold_fraction=2.0)
    coefficients_mode = numpy.polyfit(wf[0], wf[1], order)
    polynomial_mode = numpy.poly1d(coefficients_mode)
    f_pol_mode = polynomial_mode(wf[0])
    fitted_continuum_mode = numpy.array([wf[0], f_pol_mode])    
    #print 'object_spectra[1][0], wf[1][0], fitted_continuum_mode[1][0]', object_spectra[1][0], wf[1][0], fitted_continuum_mode[1][0]
    '''
    # Plot if asked to
    if plot == True:
        pyplot.title(object_name)
        pyplot.suptitle('z-corrected spectra')
        pyplot.xlabel('Wavelength [$\AA$]')
        pyplot.ylabel('Flux [ergs/s/cm$^2$/$\AA$]')    
        pyplot.plot(corr_wf[0], corr_wf[1], 'k', wf[0], wf[1], 'b', fitted_continuum[0], fitted_continuum[1], 'r')
        #pyplot.plot(mode_wf[0], mode_wf[1], 'g', fitted_continuum_mode[0],fitted_continuum_mode[1],'k--')
        pyplot.show()
        # Normalize to that continuum if norm=True
        print 'Continuum calculated. Normalization to continuum was set to: ', normalize
    if (normalize == True) and (plot == True):
        norm_flux = object_spectra[1] / f_pol
        norm_wf = numpy.array([wf[0], norm_flux])
        # Give the theoretical continuum for the line finding
        norm_continuum = theo_cont(object_spectra[0])
        pyplot.title(object_name)
        pyplot.suptitle('z-corrected spectra')
        pyplot.xlabel('Wavelength [$\AA$]')
        pyplot.ylabel('Normalized Flux')    
        pyplot.plot(norm_wf[0], norm_wf[1], 'b', norm_continuum[0], norm_continuum[1], 'r')
        pyplot.show()
        return norm_wf, norm_continuum
    else:
        return corr_wf, fitted_continuum

def test_min_chi2_of_polynomial(polynomialfit_arr, std_list, window):
    # First window
    window_lo = polynomialfit_arr[0][0]
    window_up, _ = find_nearest(polynomialfit_arr[0], window_lo+window)
    #print 'INITIAL Window: ', window_lo, window_up
    pol_flxs_lists = []
    f_win = polynomialfit_arr[1][(polynomialfit_arr[0] <= window_up)]
    pol_flxs_lists.append(f_win)
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
        f_win = polynomialfit_arr[1][(polynomialfit_arr[0] > window_lo) & (polynomialfit_arr[0] <= window_up)]
        pol_flxs_lists.append(f_win)
    chi2_list = []
    # Find the minimal chi square of the polynomial to the standard deviations of the windows
    for std,f_pol in zip(std_list, pol_flxs_lists):
        # Divide the polynomial array into the same binning as the windows
        chi2 = min_chi_square(f_pol, std)
        print 'This is Chi_square', chi2
        chi2_list.append(chi2)
    return chi2_list


############################################################################################
# LINE INFORMATION
def n4airvac_conversion(wav):
    '''This function finds the index of refraction for that wavelength.
        *** Took the equation from IAU convention for wavelength conversion described in 
            Morton (1991, ApJS, 77, 119)'''
    sigma_wav = 10000/wav
    n = 1 + 6.4328e-5 + 2.94981e-2/(146*sigma_wav*sigma_wav) + 2.5540e-4/(41*sigma_wav*sigma_wav)
    return n

def find_lines_info(object_spectra, continuum, linesinfo_file_name, text_table=False, vacuum=False):
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
    '''
    # Read the line_catalog file, assuming that the path is the same:
    # '/Users/name_of_home_directory/Documents/AptanaStudio3/science/science/spectrum/lines_catalog.txt'
    line_catalog_path = os.path.abspath('../../science/science/spectrum/lines_catalog.txt')
    #print line_catalog_path
    f = open(line_catalog_path, 'r')
    list_rows_of_file = f.readlines()
    f.close()
    wavelength = []
    element = []
    ion =[]
    forbidden = []
    how_forbidden = []
    transition = []
    strong_line = []
    for row in list_rows_of_file:
        # Disregard comment symbol
        if '#' not in row:
            # Split each row into columns
            data = row.split()
            # data[0]=wavelength, [1]=element, [2]=ion, [3]=forbidden, 
            #     [4]=how_forbidden, [5]=transition, [6]=strong_line
            wavelength.append(float(data[0]))
            element.append(data[1])
            ion.append(data[2])
            forbidden.append(data[3])
            how_forbidden.append(data[4])
            transition.append(data[5])
            strong_line.append(data[6])
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
        if sline == "nw":
            s = 5.0
        elif sline == "no":
            s = 7.0
        elif sline == "weak":
            s = 10.0
        elif sline == "medium":
            s = 15.0
        elif sline == "yes":
            s = 25.0
        elif sline == "super":
            s = 35.0
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
    # but choose the right wavelength column
    if vacuum == True:
        use_wavs = 1
        use_wavs_text = '# Used VACUUM wavelengths to find lines from line_catalog.txt'
    if vacuum == False:
        use_wavs = 0
        use_wavs_text = '# Used AIR wavelengths to find lines from line_catalog.txt'
    print 'vacuum was set to %s, %s' % (vacuum, use_wavs_text)
    for i in range(len(lines_catalog[0])):
        # find the line in the catalog that is closest to a 
        nearest2line = find_nearest_within(object_spectra[0], lines_catalog[use_wavs][i], 3)
        if nearest2line > 0.0:  
            catalog_wavs_found.append(lines_catalog[use_wavs][i])
            # If the line is in the object spectra, measure the intensity and equivalent width
            # according to the strength of the line
            central_wavelength = object_spectra[0][(object_spectra[0] == nearest2line)]
            central_wavelength_list.append(float(central_wavelength))
            line_width = lines_catalog[7][i]
            lower_wav = central_wavelength - (line_width/2)
            upper_wav = central_wavelength + (line_width/2)
            width_list.append(line_width)
            F, C = get_net_fluxes(object_spectra, continuum, lower_wav, upper_wav)
            ew, lower_wav, upper_wav = EQW(object_spectra, continuum, lower_wav, upper_wav)
            continuum_list.append(float(C))
            net_fluxes_list.append(F)
            EWs_list.append(ew) 
            found_element.append(lines_catalog[2][i])
            found_ion.append(lines_catalog[3][i])
            found_ion_forbidden.append(lines_catalog[4][i])
            found_ion_how_forbidden.append(lines_catalog[5][i])
    # Create the table of Net Fluxes and EQWs
    if text_table == True:
        #linesinfo_file_name = raw_input('Please type name of the .txt file containing the line info. Use the full path.')
        txt_file = open(linesinfo_file_name, 'w+')
        print >> txt_file,  use_wavs_text
        print >> txt_file,   '# Positive EW = emission        Negative EW = absorption' 
        print >> txt_file,  ('{:<12} {:<12} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16} {:>16} {:>12}'.format('Catalog WL', 'Observed WL', 'Element', 'Ion', 'Forbidden', 'How much', 'Width[A]', 'Flux [cgs]', 'Continuum [cgs]', 'EW [A]'))
        for cw, w, e, i, fd, h, s, F, C, ew in zip(catalog_wavs_found, central_wavelength_list, found_element, found_ion, found_ion_forbidden, found_ion_how_forbidden, width_list, net_fluxes_list, continuum_list, EWs_list):
            print >> txt_file,  ('{:<12.3f} {:<12.3f} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16.3e} {:>16.3e} {:>12.3f}'.format(cw, w, e, i, fd, h, s, F, C, ew))
        txt_file.close()
        print 'File   %s   writen!' % linesinfo_file_name
    elif text_table == False:
        print '# Positive EW = emission        Negative EW = absorption' 
        print ('{:<12} {:<12} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16} {:>16} {:>12}'.format('Catalog WL', 'Observed WL', 'Element', 'Ion', 'Forbidden', 'How much', 'Width[A]', 'Flux [cgs]', 'Continuum [cgs]', 'EW [A]'))
        for cw, w, e, i, fd, h, s, F, C, ew in zip(catalog_wavs_found, central_wavelength_list, found_element, found_ion, found_ion_forbidden, found_ion_how_forbidden, width_list, net_fluxes_list, continuum_list, EWs_list):
            print ('{:<12.3f} {:<12.3f} {:>12} {:<12} {:<12} {:<12} {:<12} {:>16.3e} {:>16.3e} {:>12.3f}'.format(cw, w, e, i, fd, h, s, F, C, ew))
    return catalog_wavs_found, central_wavelength_list, width_list, net_fluxes_list, continuum_list, EWs_list


def get_net_fluxes(object_spectra, continuum, lower_wav, upper_wav):
    '''
    This function finds the integrated flux of the line given by the lower and upper
    wavelengths.
    REQUIREMENTS:
    # object_spectra = the 2D array of wavelength and flux
    # continuum = the 2D array of wavelength and flux for the continuum
    # lower_wav, upper_wav = limits of integration
    FUNCTION RETURNS:
    # the net flux and corresponding continuum of the integration between lower and upper wavelengths
    '''
    net_flux = object_spectra[1][(object_spectra[0] >= lower_wav) & (object_spectra[0] <= upper_wav)]
    F = sum(net_flux)
    #print object_spectra[0][(object_spectra[0] >= lower_wav) & (object_spectra[0] <= upper_wav)], F
    net_continua = continuum[1][(continuum[0] >= lower_wav) & (continuum[0] <= upper_wav)]
    C = sum(net_continua) / len(net_continua)
    #C = numpy.median(net_continua) gives the same as the average value
    return F, C


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
    eqw = sum(difference) * dlambda* (-1)   # the -1 is because of the definition of EQW
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

def EQW(data_arr, cont_arr, lower, upper):
    '''
    This function the EW integrating over the interval given by the lower and upper limits.
    *** THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    # data_arr = data array that contains both wavelength and flux
    # cont_arr = continuum array that also contains both wavelength and flux
    # ARRAYS MUST BE SAME DIMENSIONS
    # lower = where the integration should begin
    # upper = final point of the integration
    # THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    '''
    # Finding closest wavelength to the desired lower and upper limits
    lolim, _ = find_nearest(data_arr[0], lower)
    uplim, _ = find_nearest(data_arr[0], upper)
    #print('Closest points in array to lower limit and upper limit: %f, %f' % (lolim, uplim))
    #width = uplim - lolim
    #print('Actual width = %f' % (width))
    # Finding the line arrays to use in the integration
    wavelength, flux = selection(data_arr[0], data_arr[1], lolim, uplim)
    w_cont, flux_cont = selection(cont_arr[0], cont_arr[1], lolim, uplim)
    # Interpolate if the flux selection array is empty 
    if len(flux_cont) == 0:
        x_list = []
        y_list = []
        for i in range(len(flux)):
            x = wavelength[i]
            y = numpy.interp(x, cont_arr[0], cont_arr[1])
            x_list.append(x)
            y_list.append(y)
        w_cont = numpy.array(x_list)
        flux_cont = numpy.array(y_list)
    # In case arrays do not have the exact same wavelengths, I am removing the second to last element
    # so that the width remains the same.
    object_selection = numpy.array([wavelength, flux])
    continuum_selection = numpy.array([w_cont, flux_cont])
    rows = len(wavelength)
    object_selection, continuum_selection, _, _ = rebin_to_desired_rows(object_selection, continuum_selection, rows)
    w = object_selection[0]
    f = object_selection[1]
    f_cont = continuum_selection[1]
    # Finding the average step for the integral
    N = len(w)
    i = 0
    diffs_list = []
    for j in range(1, N):
        point_difference = w[j] - w[i]
        diffs_list.append(point_difference)
        i = i + 1
    dlambda = sum(diffs_list) / float(N)
    # Actually solving the eqw integral
    difference = 1 - (f / f_cont)
    eqw = sum(difference) * dlambda * (-1)   # the -1 is because of the definition of EQW    
    return (eqw, lolim, uplim)

def EQW_iter(data_arr, cont_arr, line, guessed_width=2.0):
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

def EQW_initial_guess(data_arr, cont_arr, line, Teff, guessed_EQW=1):
    '''
    # line = line rest wavelength of the line of interest
    # data_arr = data array that contains both wavelength and flux
    # cont_arr = continuum array that also contains both wavelength and flux
    # Teff = effective temperature of the star to plug into eqn 1 of Valls-Gabaud (1993).
    #    * Teff is ONLY needed if expected_EQW is calculated
    # guessed_EQW = initial guess of the EQW according to Valls-Gabaud (1993)
    
    #***** data_arr  and  cont_arr  HAVE to have the same dimensions
    
    # THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    '''
    if guessed_EQW == 1:
        expected_eqw = -420.0 * numpy.exp(-1*float(Teff)/6100.0) 
    else: 
        expected_eqw = guessed_EQW    
    print('expected_eqw = %f' % expected_eqw)
    
    lolim = line - numpy.fabs(expected_eqw/2.0)
    uplim = line + numpy.fabs(expected_eqw/2.0)
    print('lolim, uplim', lolim, uplim)
    guessed_width = uplim - lolim
    test_eqw, test_lolim, test_uplim = EQW_iter(data_arr, cont_arr, line, guessed_width)
    return(test_eqw, test_lolim, test_uplim)
    
    
def EQW_initial_guessVG(data_arr, cont_arr, line, Teff, guessed_EQW=1):
    '''
    # line = line rest wavelength of the line of interest
    # data_arr = data array that contains both wavelength and flux
    # cont_arr = continuum array that also contains both wavelength and flux
    # Teff = effective temperature of the star to plug into eqn 1 of Valls-Gabaud (1993).
    #    * Teff is ONLY needed if expected_EQW is calculated
    # guessed_EQW = initial guess of the EQW according to Valls-Gabaud (1993)
    
    #***** data_arr  and  cont_arr  HAVE to have the same dimensions
    
    # THE DEFINITION OF EQW USED IS POSITIVE FOR EMISSION AND NEGATIVE FOR ABSORPTION
    '''
    if guessed_EQW == 1:
        VG_eqw_object = -420.0 * numpy.exp(-1*float(Teff)/6100.0) 
    else: 
        VG_eqw_object = guessed_EQW    
    print('Valls-Gabaud eqw = %f' % VG_eqw_object)
    
    test_eqw, test_lolim, test_uplim = EQW_iter(data_arr, cont_arr, line, guessed_width=2.0)
    interval_length = numpy.fabs(VG_eqw_object) / 2.0
    upper_allowed_eqw = numpy.fabs(VG_eqw_object)+(interval_length/2.0)
    lower_allowed_eqw = numpy.fabs(VG_eqw_object)-(interval_length/2.0)
    #print('lower_allowed_eqw, upper_allowed_eqw', lower_allowed_eqw, upper_allowed_eqw)
    iteration = 0
    increase_width = 2.0
    conditions_met = 0
    if (numpy.fabs(test_eqw) <= upper_allowed_eqw) and (numpy.fabs(test_eqw) >= lower_allowed_eqw):
        conditions_met = 1       
    
    while conditions_met != 1:        
        #print('*** in the loop: expected_eqw=%f, test_eqw=%f, numpy.fabs(test_eqw)=%f' % (VG_eqw_object, test_eqw, numpy.fabs(test_eqw)))
        #print('iteration number = %i' % iteration)
        new_lolim = test_lolim - increase_width/2.0
        new_uplim = test_uplim + increase_width/2.0
        #print('new_lolim, new_uplim', new_lolim, new_uplim)
        new_guessed_width = new_uplim - new_lolim
        #test_eqw, test_lolim, test_uplim = EQW(data_arr, cont_arr, new_lolim, new_uplim)
        test_eqw, test_lolim, test_uplim = EQW_iter(data_arr, cont_arr, line, new_guessed_width)
        if (numpy.fabs(test_eqw) <= upper_allowed_eqw) and (numpy.fabs(test_eqw) >= lower_allowed_eqw):
            conditions_met = 1
        else:
            iteration = iteration + 1
    print('iteration number = %i' % iteration)
    final_width = test_uplim - test_lolim
    print('Final array-with of expected_eqw = %f' % (final_width))
    return(VG_eqw_object, test_eqw, test_lolim, test_uplim)

#### Full width half maximum 
def FWHM(sig):
    # This can only be runned after the gaus_fit function
    fwhm = 2 * (2 * math.log1p(2))**0.5 * sig
    print ('FWHM = ', fwhm)
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
    std = ( 1/(N-1) * sum(diff2meansq_list) )**(0.5)
    #print 'sigma = ', std
    return std

def absolute_err(measurement, true_value):
    '''This function finds the absolute value of the error given the measurement and true values.'''
    abs_err = numpy.fabs(numpy.fabs(measurement) - numpy.fabs(true_value))
    return abs_err

def relative_err(measurement, true_value):
    '''This function finds the fractional error given the measurement and true values. It returns a percentage.'''
    rel_err = ( absolute_err(measurement, true_value) / numpy.fabs(true_value) ) * 100
    return rel_err

def min_chi_square(arr, estimated):
    '''Test if the observed values are close enough to the expected value.'''
    chi_square_list = []
    for observed in arr:            
        chi_sq = ((observed - estimated) * (observed - estimated)) / estimated
        chi_square_list.append(chi_sq)
    min_chi_sq = sum(chi_square_list)
    return min_chi_sq


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
    diff = numpy.fabs(choped_arr - value)
    diff_min = min(diff)
    return float(choped_arr[diff==diff_min])

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