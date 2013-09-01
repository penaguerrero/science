import numpy
import math
import os
import logging
from scipy import ndimage
from pprint import pprint

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

def rebin(arr, factor):
    ''' arr: array-like tuple
    factor: rebin_factor tuple'''
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

def get_factors_and_rebin(spectrum_arr, continuum_arr, desired_rows=500):
    '''
    ### THIS FUNCTIONS DOES EVERYTHING AT ONCE FOR SPECTRA WITH SPECTRA AND CONT ARRAYS OF DIFFERENT DIMENSIONS
    # spectrum_arr = numpy array of wavelength and flux
    # continuum_arr = numpy array of wavelength and flux
    # desired_rows = number of lines for output file, by default it is 600
    # THIS FUNCTION RETURNS: 
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

def resolving_power(line, arr):
    '''This function determines the resolving power R at the line wavelength.
    # arr must be a numpy array of wavelength and flux. '''
    closest_line, idxline = find_nearest(arr[0,:], line)
    #diffLyA1 = arr[0, idxline+1] - closest_line
    #diffLyA2 = closest_line - arr[0, idxline-1]
    #new_delta_lambda = (diffLyA1 + diffLyA2) / 2.0
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
        
    if small_arr_shape[1] > desired_rows:
        small_arr_factor = float(desired_rows) / float(small_arr_shape[1])
    elif small_arr_shape[1] < desired_rows:
        small_arr_factor = float(desired_rows) / float(small_arr_shape[1])
    
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

def find_nearest(arr, value):
    idx=(numpy.abs(arr-value)).argmin()
    return arr[idx], idx

def closest(thelist, value) :
    return min((abs(value - i), i) for i in thelist)[1]
    
def find_index_in_list(the_list, item):
    for idx in range(0, len(the_list)):
        if the_list[idx] == item:
            return(idx)
        
'''def findinlist(func, thelist):
    # Returns the first item in the list where function(item) == True.
    for item in thelist:
        if func(item):
            return item
'''        
def midpoint(p1, p2):
    return p1 + (p2-p1)/2

def theo_cont(wave_arr, scale_factor=1.0):
    # Since these are theoretical data, the continuum is by definition at 1 when normalizing
    # the data array has to be the wavelength element, the result is going to be an array
    # of both wavelength and flux.
    cont_temp = []
    for _ in wave_arr:
        cont_temp.append(1.0 * scale_factor)
    theoretical_cont = numpy.array([wave_arr, cont_temp]) 
    return theoretical_cont

def findXinY (Xarr, Yarr, value):
    # This function finds value in array Y and returns the corresponding value in array X
    # Wanted value is in array Xarr
    # Given value is in array Yarr
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

def EQW_line_fixed(line_arr, line_cont, line, width=10.0):
    '''
    # line is the rest wavelength of the line of interest
    # line_arr is the tuple array of wavelength and flux for the line
    # line_cont is also a tuple of wavelength and flux (=1 if it is theoretical)
    '''
    # Limits to find the EWQ
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
    wavelength, flux = selection(data_arr[0, :], data_arr[1, :], lolim, uplim)
    _, flux_cont = selection(cont_arr[0, :], cont_arr[1, :], lolim, uplim)
    # In case arrays do not have the exact same wavelengths, I am removing the second to last element
    # so that the width remains the same.
    if len(flux) > len(flux_cont):
        wavelength = numpy.delete(wavelength, wavelength[len(wavelength)-2])
        flux = numpy.delete(flux, flux[len(flux)-2])
    elif len(flux) < len(flux_cont):       
        flux_cont = numpy.delete(flux_cont, flux_cont[len(flux_cont)-2])
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
    return (eqw, lolim, uplim)

def EQW_iter(data_arr, cont_arr, line, guessed_width=2.0):
    '''
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
    
    
    
'''    if (numpy.fabs(test_eqw) <= upper_allowed_eqw) and (numpy.fabs(test_eqw) >= lower_allowed_eqw):
        return(VG_eqw_object, test_eqw, test_lolim, test_uplim)        
    
    last_wavelength = data_arr[0, len(data_arr[0])-1]
    iteration = 0
    increase_width = 2.0
    while test_uplim <= last_wavelength:        
        print('*** in the loop: expected_eqw=%f, test_eqw=%f, numpy.fabs(test_eqw)=%f' % (VG_eqw_object, test_eqw, numpy.fabs(test_eqw)))
        print('iteration number = %i' % iteration)
        new_lolim = test_lolim - increase_width/2.0
        new_uplim = test_uplim + increase_width/2.0
        print('new_lolim, new_uplim', new_lolim, new_uplim)
        new_guessed_width = new_uplim - new_lolim
        #test_eqw, test_lolim, test_uplim = EQW(data_arr, cont_arr, new_lolim, new_uplim)
        test_eqw, test_lolim, test_uplim = EQW_iter(data_arr, cont_arr, line, new_guessed_width)
        if (numpy.fabs(test_eqw) <= upper_allowed_eqw) and (numpy.fabs(test_eqw) >= lower_allowed_eqw):
            break
        else:
            iteration = iteration + 1
    final_width = test_uplim - test_lolim
    print('Final array-with of expected_eqw = %f' % (final_width))
    return(VG_eqw_object, test_eqw, test_lolim, test_uplim)
'''



'''### THE FOLLOWING 3 FUNCTIONS WORK TOGETHER TO FIND THE BEST GAUSSIAN FIT TO THE LINE ###'''
'''1'''
def model(x, continuum, coeffs):    
    # x is the variable
    # continuum    noise
    # coeffs=[0]   amplitude
    # coeffs=[1]   mean or center
    # coeffs[2]    width or std deviation
    return continuum - coeffs[0] * numpy.exp( - ((x-coeffs[1])/coeffs[2])**2 )
'''2'''
def residuals(coeffs, continuum, y, x):
    return y - model(x, continuum, coeffs)
'''3'''
def gaus_fit(line_arr, line_cont, amplitude, center, sig):
    # The line array and continuum array must be tuples of wavelength and flux
    # sig=coeff[2] is the sigma, width, or standard deviation
    # The actual equation for a gaussian curve
    gaussian = lambda x, continuum: continuum + amplitude * numpy.exp(-((x-center)/sig)**2)
    gaussian_fit = gaussian(line_arr[0], line_cont[1])
    #print('The gaussian fit is', gaussian_fit)
    return (gaussian_fit)

'''
# THIS FUNCTION WORKS BUT DOES NOT GIVE THE CORRECT FIT
def gaus_fit(line_arr):
    # This function can be runned after the find_line function
    # The line array must be a tuple of wavelength and flux
    mean_x = sum(line_arr[0])/(line_arr[0].size)
    sig = math.sqrt(abs(sum((line_arr[0] - mean_x)**2 * line_arr[0])/sum(line_arr[0]))) 
    # sig is the sigma, width, or standard deviation
    cent_y = line_arr[1].max()
    cent_x = line_arr[0][line_arr[1]==cent_y]
    center = (cent_x, cent_y)
    # The actual equation for a gaussian curve
    gaussian = lambda t : cent_y * ( 1/(sig*((2*math.pi)**0.5)) * numpy.exp(-((t-mean_x)/sig)**2) )
    gaussian_fit = gaussian(line_arr[0])
    print('The gaussian fitt is', gaussian_fit)
    return (gaussian_fit, sig, center)
'''

def FWHM(sig):
    # This can only be runned after the gaus_fit function
    fwhm = 2 * (2 * math.log1p(2))**0.5 * sig
    print ('FWHM = ', fwhm)
    return fwhm

def factor_newshape(continuum_arr, lineintensity_arr, lineintensity_factor, desired_rows=500):
    percent = 1.00
    oldshape = lineintensity_arr.shape
    newshape = lineintensity_arr.shape
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
    print(newshape)
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
        print 'Arrays are the same dimensions. Factor is 1.0'
        return factor_same
    small_arr_columns, small_arr_rows = numpy.shape(small_arr)
    big_arr_columns, big_arr_rows = numpy.shape(big_arr)
    # The number of colums in the small array has to be equal to the columns in line big array
    if big_arr_columns != small_arr_columns:
        raise ValueError('Number of columns in files do not match')
    factor_same = (big_arr_rows/small_arr_rows)**0.5 
    return factor_same

def get_factor_lineintensity(continuum_arr, lineintensity_arr):
    ''' ### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRUM ARRAYS ARE NOT SAME DIMENSIONS'''
    factor_same = same_rebin_factor(continuum_arr, lineintensity_arr)
    _, continuum_rows = continuum_arr.shape
    lineintensity_columns, lineintensity_rows = lineintensity_arr.shape
    factor_lineintensity = (continuum_rows * factor_same) / lineintensity_rows
    return (lineintensity_columns, factor_lineintensity)

def get_factor_continuum(continuum_arr, lineintensity_arr):
    '''### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRUM ARRAYS ARE NOT SAME DIMENSIONS'''
    factor_continuum = same_rebin_factor(continuum_arr, lineintensity_arr)
    continuum_columns, _ = continuum_arr.shape
    #print('This is the factor_eq for cont', factor_continuum)
    return (continuum_columns, factor_continuum)
    
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

class spectrum:
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