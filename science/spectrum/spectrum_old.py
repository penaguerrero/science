import numpy
import math
from scipy.optimize import fsolve
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
    return ndimage.interpolation.zoom(arr, factor)

def get_factors_and_rebin(spectrum_arr, continuum_arr):
    ### THIS FUNCTIONS DOES EVERYTHING AT ONCE FOR SPECTRA WITH SPECTRA AND CONT ARRAYS OF DIFFERENT DIMENSIONS
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
    # using those factors so that the nomber of rows is the same
    new_continuum_factor, new_lineintensity_factor = factor_newshape(continuum_arr, spectrum_arr, factors[1], desired_rows=600)
    rebin_line = rebin(spectrum_arr, (1, new_lineintensity_factor))    
    rebin_cont = rebin(continuum_arr, (1, new_continuum_factor))
    return rebin_line, rebin_cont, new_continuum_factor, new_lineintensity_factor

def do_rebin(spectrum_arr, continuum_arr):
    ### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRA ARRAY HAVE SAME DIMENSIONS
    orig_factors = (1, 1) # Numbers that will mumtiply colums and rows
    continuum_factor, spec_factor = factor_newshape(continuum_arr, spectrum_arr, orig_factors[1], desired_rows=500)
    rebin_spec = rebin(spectrum_arr, (1, spec_factor))    
    rebin_cont = rebin(continuum_arr, (1, continuum_factor))
    # Since arrays have same dimensions, continuum_factor = spec_factor
    return rebin_spec, rebin_cont, spec_factor    

def find_nearest(arr, value):
    idx=(numpy.abs(arr-value)).argmin()
    return arr[idx], idx

def closest(thelist, value) :
    return min((abs(value - i), i) for i in thelist)[1]
    
def midpoint(p1, p2):
    return p1 + (p2-p1)/2

def theo_cont(arr, scale_factor=1.0):
    # Since these are theoretical data, the continuum is by definition at 1 when normalizing
    # the data array has to be the wavelength element
    cont_temp = []
    for i in arr:
        cont_temp.append(1.0 * scale_factor)
    theoretical_cont = numpy.array([arr, cont_temp]) 
    return theoretical_cont

def findXinY (Xarr, Yarr, value):
    x = 0.0
    for i in range(0, len(Xarr)):
        if Yarr[i] == value:
            x = Xarr[i] 
    if x == 0.0:
        raise Exception("Value %f not found in Yarr" % (value))
        exit(1)
    return x

def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x : fun1(x) - fun2(x), x0)

def findinlist(func, thelist):
    '''Returns the first item in the list where function(item) == True. '''
    for item in thelist:
        if func(item):
            return item

def convert2abs(thelist):
    for i in range(0, len(thelist)):
        thelist[i] = abs(thelist[i])
    return thelist

def emission_max(wavelength, flux, closest_wavelength, closest_flux, tolerance=1.0):
    print('emission_max...')
    center_flux = flux.max()
    center_wavelength = findXinY(wavelength, flux, center_flux)
    while abs(center_wavelength - closest_wavelength) > tolerance:
        #print(center_wavelength, center_flux)
        flux[flux==center_flux] = 1.0e-100
        center_flux = flux.max()
        center_wavelength = findXinY(wavelength, flux, center_flux)
    return center_wavelength, center_flux

def emission_min(wavelength, flux, closest_wavelength, closest_flux, tolerance=1.0):
    print('emission min...')
    center_flux = flux.min()
    center_wavelength = findXinY(wavelength, flux, center_flux)
    while abs(center_wavelength - closest_wavelength) > tolerance:
        #print(center_wavelength, center_flux)
        flux[flux==center_flux] = 1.0e-100
        center_flux = flux.min()
        center_wavelength = findXinY(wavelength, flux, center_flux)
    return center_wavelength, center_flux

def absorption_max(wavelength, flux, closest_wavelength, closest_flux, tolerance=1.0):
    center_flux = flux.max()
    center_wavelength = findXinY(wavelength, flux, center_flux)
    while abs(center_wavelength - closest_wavelength) > tolerance:
        #print(center_wavelength, center_flux)
        flux[flux==center_flux] = 1.0e100
        center_flux = flux.max()
        center_wavelength = findXinY(wavelength, flux, center_flux)
    return center_wavelength, center_flux

def absorption_min(wavelength, flux, closest_wavelength, closest_flux, tolerance=1.0):
    center_flux = flux.min()
    center_wavelength = findXinY(wavelength, flux, center_flux)
    while abs(center_wavelength - closest_wavelength) > tolerance:
        #print(center_wavelength, center_flux)
        flux[flux==center_flux] = 1.0e100
        center_flux = flux.min()
        center_wavelength = findXinY(wavelength, flux, center_flux)
    return center_wavelength, center_flux

def find_line_intersect(x1, x2, x3, x4, y1, y2, y3, y4):
    if (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) == 0:
        raise Exception("Lines are parallel.")
        exit(1)
    else:
        Intersectx = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
        Intersecty = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4))
    return Intersectx, Intersecty

def perpendicular( a ) :
    b = numpy.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perpendicular(da)
    denom = numpy.dot( dap, db)
    num = numpy.dot( dap, dp )
    return (num / denom)*db + b1

def find_line(line, wavelength, flux, continuum, guess=5):
    # Find and gather all the wavelengths of the line
    # the cont_arr is a tuple of wavelength and flux(=1 if it is theoretical)
    # This fuction finds automatically the line width.
    # There is a limit in the line width.
    closest_wavelength, _, = find_nearest(wavelength, line)
    # Finding the actual width of the line
    center_w = closest_wavelength
    '''
    closest_flux = findXinY(flux, wavelength, closest_wavelength)
    closest_continuum, idx_closest_cont = find_nearest(continuum[0], closest_wavelength)
    print('The closest point to rest wavelength of %f is at %f, %f' % (line, closest_wavelength, closest_flux))
    x = wavelength[(wavelength > closest_wavelength-(guess/2)) & (wavelength < closest_wavelength+(guess/2))]
    y = flux[(wavelength > closest_wavelength-(guess/2)) & (wavelength < closest_wavelength+(guess/2))]   
    emission = 1
    tolerance = 1.0
    if closest_flux >= continuum[1][idx_closest_cont]:
        center_w_max, center_f_max = emission_max(x, y, closest_wavelength, closest_flux, tolerance)
        center_w_min, center_f_min = emission_min(x, y, closest_wavelength, closest_flux, tolerance)
        center_cont_max, idxcenter_cont_max = find_nearest(continuum[0], center_w_max)
        center_cont_min, idxcenter_cont_min = find_nearest(continuum[0], center_w_min)
        diff_max = abs(center_f_max - continuum[1][idxcenter_cont_max])
        diff_min = abs(center_f_min - continuum[1][idxcenter_cont_min])
        if diff_max > diff_min:
            emission = emission
            center_w = center_w_max
            center_f = center_f_max
        else:
            center_w = center_w_min
            center_f = center_f_min
            emission = 0
    # or absorption     
    if closest_flux < continuum[1][idx_closest_cont]:
        center_w_max, center_f_max = absorption_max(x, y, closest_wavelength, closest_flux, tolerance)
        center_w_min, center_f_min = absorption_min(x, y, closest_wavelength, closest_flux, tolerance)
        center_cont_max, idxcenter_cont_max = find_nearest(continuum[0], center_w_max)
        center_cont_min, idxcenter_cont_min = find_nearest(continuum[0], center_w_min)
        diff_max = abs(center_f_max - continuum[1][idxcenter_cont_max])
        diff_min = abs(center_f_min - continuum[1][idxcenter_cont_min])
        if diff_max > diff_min:
            emission = emission
            center_w = center_w_max
            center_f = center_f_max
        else:
            center_w = center_w_min
            center_f = center_f_min
            emission = 0
    print('New center:', center_w, center_f)
    '''
    w = wavelength[(wavelength > center_w-(guess/2)) & (wavelength < center_w+(guess/2))]
    f = flux[(wavelength > center_w-(guess/2)) & (wavelength < center_w+(guess/2))]
    lolim = w.min()
    uplim = w.max()
    line_width = uplim - lolim
    # finding the line continuum
    cont_w = continuum[0][(wavelength >= lolim) & (wavelength <= uplim)]
    cont_f = continuum[1][(wavelength >= lolim) & (wavelength <= uplim)]  
    # output              
    line_data = numpy.array([w, f])
    line_cont = numpy.array([cont_w, cont_f])
    print('The lower and upper limits are: ', lolim, uplim)
    print ('The final width of the line in Angstroms is', line_width)
    return (line_data, line_cont, lolim, uplim)
    
def find_line_joe(line, wavelength, flux, continuum, lowest_width=1.0, highest_width=10.0):
    # Find and gather all the wavelengths of the line
    # the cont_arr is a tuple of wavelength and flux(=1 if it is theoretical)
    # This fuction finds automatically the line width.
    # There is a limit in the line width.
    # Both lowest and highest limits are in Angstroms. Limit set so that code does not brake.
    closest_wavelength, idx_w = find_nearest(wavelength, line)
    closest_flux = findXinY(flux, wavelength, closest_wavelength)
    closest_continuum, _, = find_nearest(continuum[0], closest_wavelength)
    print('The closest point to rest wavelength of %f is at %f, %f' % (line, closest_wavelength, closest_flux))
    
    start_set = 0
    end_set = 0
    
    #for i in range(idx_w, 0, -1):
    #    if wavelength[i] < line and abs(flux[i] - continuum[1][i]) <= 0:
    #        start_set = i
    #        break

    i = idx_w
    while(True):
        if wavelength[i] < line and flux[i] >= continuum[1][i]:
            start_set = i
            break
        i -= 1
        
    i = start_set
    while(True):
        if wavelength[i] > line and flux[i] >= continuum[1][i]:
            end_set = i
            break
        i += 1

    #print('Set begins at %f (%f, %f)' % (start_set, wavelength[start_set], flux[start_set]))
    #print('Set ends at %f (%f, %f)' % (end_set, wavelength[end_set], flux[end_set]))
    
    w = wavelength[(wavelength >= wavelength[start_set]) & (wavelength <= wavelength[end_set])]
    f = flux[(wavelength >= wavelength[start_set]) & (wavelength <= wavelength[end_set])]
    lolim = wavelength[start_set]
    uplim = wavelength[end_set]
    cont_w = continuum[0][(wavelength >= lolim) & (wavelength <= uplim)]
    cont_f = continuum[1][(wavelength >= lolim) & (wavelength <= uplim)]
    
    line_width =  uplim - lolim
    print('my first estimate of lower and upper limits', lolim, uplim)
    print('so the first line width estimate is', line_width)
    
    if line_width < lowest_width:
        print('ENTERED THE WILL NOT FAIL PART')
    
        width = lowest_width
        x = wavelength[(wavelength > closest_wavelength-(width/2)) & (wavelength < closest_wavelength+(width/2))]
        y = flux[(wavelength > closest_wavelength-(width/2)) & (wavelength < closest_wavelength+(width/2))]
        # Again check if emission
        if closest_flux > findXinY(continuum[1], continuum[0], closest_continuum):
            new_center_y = y.max()
            print('Actual line width='+repr(line_width)+' < minimum width='+repr(width))   
        # or absorption     
        if closest_flux < findXinY(continuum[1], continuum[0], closest_continuum):
            new_center_y = y.min()
            print('Actual line width='+repr(line_width)+' < minimum width='+repr(width))   
        new_center_x = findXinY(x, y, new_center_y)
        #print('Closest point was above continuum but line width<3.')
        print('The peak of the line is at', new_center_x, new_center_y)
        # Finding the actual width of the line
        w = wavelength[(wavelength > new_center_x-(width/2)) & (wavelength < new_center_x+(width/2))]
        f = flux[(wavelength > new_center_x-(width/2)) & (wavelength < new_center_x+(width/2))]
        lolim = w.min()
        uplim = w.max()
        line_width = uplim - lolim
        print('my second estimate of lower and upper limits', lolim, uplim)
        print('so the first line width estimate is', line_width)
        # finding the line continuum
        cont_w = continuum[0][(wavelength >= lolim) & (wavelength <= uplim)]
        cont_f = continuum[1][(wavelength >= lolim) & (wavelength <= uplim)]
                
    line_data = numpy.array([w, f])
    line_cont = numpy.array([cont_w, cont_f])
    print('The lower and upper limits are: ', lolim, uplim)
    print ('The final width of the line in Angstroms is', line_width)
    return (line_data, line_cont, lolim, uplim)
        
    
def find_line_old(line, wav_arr, flx_arr, cont_arr, guess=5, lowest_width=1.0, highest_width=10.0):
    # Find and gather all the wavelengths of the line
    # the cont_arr is a tuple of wavelength and flux(=1 if it is theoretical)
    # This fuction finds automatically the line width.
    # Unless otherwise commanded, the line has an initial guess width of 5 Angstroms wide by default,
    # however, there is a limit in the line width.
    # Both lowest and highest limits are in Angstroms. Limit set so that code does not brake.
    closest_x, idx_x = find_nearest(wav_arr, line)
    closest_y = flx_arr[idx_x] #findXinY(flx_arr, wav_arr, closest_x)
    print('The closest point to rest wavelength of ', line,' is at ', closest_x, closest_y)
    x = wav_arr[(wav_arr > closest_x-(guess/2)) & (wav_arr < closest_x+(guess/2))]
    y = flx_arr[(wav_arr > closest_x-(guess/2)) & (wav_arr < closest_x+(guess/2))]
    print("x = %s , y = %s" % (type(x), type(y)))
    print('initial arrays', x, y)
    # Closest wavelength to closest_x in continuum array
    cl_cont, idx_cl_cont = find_nearest(cont_arr[0], closest_x)
    # Figure out if the line is in emission
    emission = 1
    tolerance = 1.0
    temp_y = y
    if closest_y >= cont_arr[1][idx_cl_cont]:
        new_center_y = temp_y.max()
        new_center_x = findXinY(x, y, new_center_y)
        while abs(new_center_x - closest_x) > tolerance:
            print('in the loop')
            print(new_center_x, new_center_y)
            temp_y[temp_y==new_center_y] = 1.0e-100
            new_center_y = temp_y.max()
            new_center_x = findXinY(x, temp_y, new_center_y)

        emission = emission
        print('New center above the continuum')
    # or absorption     
    if closest_y < findXinY(cont_arr[1], cont_arr[0], cl_cont):
        new_center_y = temp_y.min()
        new_center_x = findXinY(x, y, new_center_y)        
        while abs(new_center_x - closest_x) > tolerance:
            print('in the loop')
            print(new_center_x, new_center_y)
            temp_y[temp_y==new_center_y] = 1.0e100
            new_center_y = temp_y.min()
            new_center_x = findXinY(x, y, new_center_y)
        emission = 0
        print('New center below the continuum')
    #print('Closest point in cont array to line is', cl_cont, cont_arr[1][cont_arr[0]==cl_cont])
    #####
    print('New center:', new_center_x, new_center_y)
    # Finding the actual width of the line
    initial_x = wav_arr[(wav_arr > new_center_x-(guess/2)) & (wav_arr < new_center_x+(guess/2))]
    initial_y = flx_arr[(wav_arr > new_center_x-(guess/2)) & (wav_arr < new_center_x+(guess/2))]
    initial_data = numpy.array([initial_x, initial_y])
    print('got initial arrays')
    pprint(initial_data)
    # finding the line continuum
    lo = min(initial_x)
    up = max(initial_x)
    print('Initial lo and up wavelengths', lo, up)
    ini_cont_x = cont_arr[0][(wav_arr >= lo) & (wav_arr <= up)]
    ini_cont_y = cont_arr[1][(wav_arr >= lo) & (wav_arr <= up)]

    diffy =[]
    for i in range(0, len(initial_y)):
        diffy.append(initial_y[i] - ini_cont_y[i])
    print('diffs', diffy)
    
    if emission == 1.0:
        idx_center = diffy.index(max(diffy))
    else:
        idx_center = diffy.index(min(diffy))
        
    diffy_left_em = []
    diffy_left_ab = []
    diffx_left_em = []
    diffx_left_ab = []
    for ii in range(0, idx_center):
        if diffy[ii] > 0.0:
            diffy_left_em.append(diffy[ii])
            diffx_left_em.append(new_center_x - initial_x[ii])
        else:
            diffy_left_ab.append(diffy[ii])
            diffx_left_ab.append(new_center_x - initial_x[ii])            
    
    diffy_right_em = []
    diffy_right_ab = []
    diffx_right_em = []
    diffx_right_ab = []
    for iii in range(idx_center, len(initial_y)):
        if diffy[ii] > 0.0:
            diffy_right_em.append(diffy[iii])
            diffx_right_em.append(initial_x[iii] - new_center_x)
        else: #if diffy[iii] < 0.0:
            diffy_right_ab.append(diffy[iii])
            diffx_right_ab.append(initial_x[iii] - new_center_x)
   
    print('figured out positives and negatives')
        
    if emission == 1:
        print('Line in emission')
        diffx_left = convert2abs(diffx_left_em) 
        diffy_left = convert2abs(diffy_left_em)
        diffx_right = convert2abs(diffx_right_em)
        diffy_right = convert2abs(diffy_right_em)
        print('left diffx', diffx_left)
        print('left diffy', diffy_left)
        print('right diffx', diffx_right)
        print('right diffy', diffy_right)
    if emission == 0:    
        print('Line in absorption')    
        diffx_left = convert2abs(diffx_left_ab) 
        diffy_left = convert2abs(diffy_left_ab)
        diffx_right = convert2abs(diffx_right_ab)
        diffy_right = convert2abs(diffy_right_ab)
        print('left diffx', diffx_left)
        print('left diffy', diffy_left)
        print('right diffx', diffx_right)
        print('right diffy', diffy_right)
    '''
    # LEFT of peak
    if (len(diffx_left) > 2):
        peak1x = diffx_left.index(min(diffx_left))
        diffx_left.pop(peak1x)
    if (len(diffy_left) > 2):
        peak1y = diffy_left.index(max(diffy_left))
        diffy_left.pop(peak1y)
    # RIGHT of peak
    if (len(diffx_right) > 2):
        peak2x = diffx_right.index(min(diffx_right))
        diffx_right.pop(peak2x)
    if (len(diffy_right) > 2):
        peak2y = diffy_right.index(max(diffy_right))
        diffy_right.pop(peak2y)
    
    print('whats left over', diffx_left)
    print('whats left over', diffy_left)
    print('whats left over', diffx_right)
    print('whats left over', diffy_right)
    '''
    
    idx_lolimx, _, = min(enumerate(diffx_left), key=lambda x: abs(x[1]-0.0))
    idx_lolimy, lolim_in_diffy = min(enumerate(diffy_left), key=lambda x: abs(x[1]-0.0))
    
    while idx_lolimy != idx_lolimx:
        diffx_left[idx_lolimx] = 1.0e100
        diffy_left[idx_lolimy] = 1.0e100
        idx_lolimx, _, = min(enumerate(diffx_left), key=lambda x: abs(x[1]-0.0))
        idx_lolimy, lolim_in_diffy = min(enumerate(diffy_left), key=lambda x: abs(x[1]-0.0))
    idx_lolim = min(range(len(diffy)), key=lambda i: abs(diffy[i]-lolim_in_diffy))
    lolim = initial_x[idx_lolim]
    
    idx_uplimx, _, = min(enumerate(diffx_right), key=lambda x: abs(x[1]-0.0))    
    idx_uplimy, uplim_in_diffy = min(enumerate(diffy_right), key=lambda x: abs(x[1]-0.0))    
    while idx_uplimy != idx_lolimx:
        diffx_right[idx_uplimx] = 1.0e100
        diffy_right[idx_uplimy] = 1.0e100
        idx_uplimx, _, = min(enumerate(diffx_right), key=lambda x: abs(x[1]-0.0))
        idx_uplimy, uplim_in_diffy = min(enumerate(diffy_right), key=lambda x: abs(x[1]-0.0))  
    idx_uplim = min(range(len(diffy)), key=lambda i: abs(diffy[i]-uplim_in_diffy))
    uplim = initial_x[idx_uplim]    
    
    print('Actual lolim and uplim', lolim, uplim)
    line_width = uplim - lolim            
    
    condition = (initial_x >= lolim) & (initial_x <= uplim)
    line_x = initial_x[condition]
    line_y = initial_y[condition]
    cont_x = ini_cont_x[condition]
    cont_y = ini_cont_y[condition]
    line_data = numpy.array([line_x, line_y])
    line_cont = numpy.array([cont_x, cont_y])
    
    # SEARCHES FOR LOWEST LIMIT SO THAT CODE DOES NOT BRAKE AND, IF DIVIDES WIDE LINES ACCORDING TO THE HIGHEST WIDTH DEFINED. 
    if line_width < lowest_width:
        width = lowest_width
        x = wav_arr[(wav_arr > closest_x-(width/2)) & (wav_arr < closest_x+(width/2))]
        y = flx_arr[(wav_arr > closest_x-(width/2)) & (wav_arr < closest_x+(width/2))]
        # Again check if emission
        if closest_y > findXinY(cont_arr[1], cont_arr[0], cl_cont):
            new_center_y = y.max()
            print('Actual line width='+repr(line_width)+' < minimum width='+repr(width))   
        # or absorption     
        if closest_y < findXinY(cont_arr[1], cont_arr[0], cl_cont):
            new_center_y = y.min()
            print('Actual line width='+repr(line_width)+' < minimum width='+repr(width))   
        new_center_x = findXinY(x, y, new_center_y)
        #print('Closest point was above continuum but line width<3.')
        print('The peak of the line is at', new_center_x, new_center_y)
        # Finding the actual width of the line
        line_x = wav_arr[(wav_arr > new_center_x-(width/2)) & (wav_arr < new_center_x+(width/2))]
        line_y = flx_arr[(wav_arr > new_center_x-(width/2)) & (wav_arr < new_center_x+(width/2))]
        line_data = numpy.array([line_x, line_y])
        lolim = min(line_x)
        uplim = max(line_x)
        line_width = uplim - lolim
        # finding the line continuum
        line_cont_x = cont_arr[0][(wav_arr >= lolim) & (wav_arr <= uplim)]
        line_cont_y = cont_arr[1][(wav_arr >= lolim) & (wav_arr <= uplim)]
        line_cont = numpy.array([line_cont_x, line_cont_y])
    if line_width > highest_width:
        width = highest_width
        x = wav_arr[(wav_arr > closest_x-(width/2)) & (wav_arr < closest_x+(width/2))]
        y = flx_arr[(wav_arr > closest_x-(width/2)) & (wav_arr < closest_x+(width/2))]
        # Again check if emission
        if closest_y > findXinY(cont_arr[1], cont_arr[0], cl_cont):
            new_center_y = y.max()
            print('Actual line width='+repr(line_width)+' < minimum width='+repr(width))   
        # or absorption     
        if closest_y < findXinY(cont_arr[1], cont_arr[0], cl_cont):
            new_center_y = y.min()
            print('Actual line width='+repr(line_width)+' < minimum width='+repr(width))  
        new_center_x = findXinY(x, y, new_center_y)
        #print('Closest point was above continuum but line width<3.')
        print('The peak of the line is at', new_center_x, new_center_y)
        # Finding the actual width of the line
        line_x = wav_arr[(wav_arr > new_center_x-(width/2)) & (wav_arr < new_center_x+(width/2))]
        line_y = flx_arr[(wav_arr > new_center_x-(width/2)) & (wav_arr < new_center_x+(width/2))]
        line_data = numpy.array([line_x, line_y])
        lolim = min(line_x)
        uplim = max(line_x)
        line_width = uplim - lolim
        # finding the line continuum
        line_cont_x = cont_arr[0][(wav_arr >= lolim) & (wav_arr <= uplim)]
        line_cont_y = cont_arr[1][(wav_arr >= lolim) & (wav_arr <= uplim)]
        line_cont = numpy.array([line_cont_x, line_cont_y])
    print('The lower and upper limits are: ', lolim, uplim)
    print ('The final width of the line in Angstroms is', line_width)
    return (line_data, line_cont, lolim, uplim)
    
def EQW(line_arr, line_cont, lolim, uplim):
    # line is the rest wavelength of the line of interest
    # line_arr is the tuple array of wavelength and flux for the line
    # line_cont is also a tuple of wavelength and flux (=1 if it is theoretical)
    # Finding the average step for the integral
    #next_uplim, _ = find_nearest(line_arr, uplim)
    #dlambda = uplim - next_uplim
    N = len(line_arr[0])
    dlambda = abs((line_arr[0][N-2]-lolim)/(uplim-lolim))
    # Actually solving the eqw integral
    #difference = 1-(line_arr[1]/line_cont[1])
    #eqw = sum(difference) * dlambda

    # TRANSLATION OF PYSPECKIT
    difference = line_cont[1] - line_arr[1]
    sumofdiff = sum(difference) * dlambda
    eqw = sumofdiff / numpy.median(line_cont[1])    
    
    eqw_min = midpoint(lolim, uplim) - (abs(eqw) / 2.0)
    eqw_max = midpoint(lolim, uplim) + (abs(eqw) / 2.0)
    return (eqw, eqw_min, eqw_max)

# THE FOLLOWING 3 FUNCTIONS WORK TOGETHER TO FIND THE BEST GAUSSIAN FIT TO THE LINE
def model(x, continuum, coeffs):
    # x is the variable
    # continuum    noise
    # coeffs=[0]   amplitude
    # coeffs=[1]   mean or center
    # coeffs[2]    width or std deviation
    return continuum - coeffs[0] * numpy.exp( - ((x-coeffs[1])/coeffs[2])**2 )
def residuals(coeffs, continuum, y, x):
    return y - model(x, continuum, coeffs)
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

def factor_newshape(continuum_arr, lineintensity_arr, lineintensity_factor, desired_rows=600):
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
    print("Percent: %f  Newshape: %s  New Line Intensity Factor: %f" % (percent, newshape, new_lineintensity_factor))
    print(newshape)
    return(new_continuum_factor, new_lineintensity_factor)

"""
def factor_newshape(continuum_arr, lineintensity_arr, continuum_factor): #compares shapes until a common shape is discovered
    percent = 1.00
    newshape = continuum_arr.shape
    continuum_newfactor = 0
    while percent > 0.00:
        if newshape[1] == 600:
            break
        newcontinuum_factor = continuum_factor * percent
        temp_continuum_arr = rebin(continuum_arr, (1, newcontinuum_factor))
        newshape = temp_continuum_arr.shape
        print(newshape)
        percent -= 0.01
    #lineintensityshape = lineintensity_arr.shape
    newlineintensity_factor = 1/newcontinuum_factor # = (newshape[1] * newcontinuum_factor) / lineintensityshape[1]
    print('The new shape', newshape)
    return (newcontinuum_factor, newlineintensity_factor)
"""

def get_factor_continuum(continuum_shape, lineintensity_shape):
    ### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRUM ARRAYS ARE NOT SAME DIMENSIONS
    continuum_columns, continuum_rows = continuum_shape.shape
    lineintensity_columns, lineintensity_rows = lineintensity_shape.shape
    # The number of colums in the continuum has to be equal to the columns in line intensity
    if lineintensity_columns != continuum_columns:
        raise ValueError('Number of columns do not match')
    factor_same = (lineintensity_rows/continuum_rows)**0.5 # this is the factor that rebins _fin and _cont by the same amount
    factor_continuum = factor_same 
    print('This is the factor_eq for cont', factor_continuum)
    return (continuum_columns, factor_continuum)

def get_factor_lineintensity(continuum_shape, lineintensity_shape):
    ### THIS FUNCTION IS TO BE RUNNED WHEN CONTINUUM AND SPECTRUM ARRAYS ARE NOT SAME DIMENSIONS
    continuum_columns, continuum_rows = continuum_shape.shape
    lineintensity_columns, lineintensity_rows = lineintensity_shape.shape
    # The number of colums in the continuum has to be equal to the columns in line intensity
    if lineintensity_columns != continuum_columns:
        raise ValueError('Number of columns do not match')
    factor_same = (lineintensity_rows/continuum_rows)**0.5 # this is the factor that rebins _fin and _cont by the same amount
    factor_lineintensity = (continuum_rows * factor_same) / lineintensity_rows
    return (lineintensity_columns, factor_lineintensity)
    
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
        print("hztoA...")
        A = self.hztoA(self.hz)
        print("done")
        print("jytocgs...")
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
        for i in range(0, len(w)):
            pprint( 'orig_w(%.20f) :: A_w(%.20f) => (orig_f(%.20f) :: cgs_f(%.20f)' % 
                    (wavelength[i], w[i], flux[i], f[i]) )