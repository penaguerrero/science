import os
import shutil
import glob
import numpy
import science

from .. import spectrum
from ..claus import clausfile

def into2cols(path_CMFGEN, path_results, lower_wave, upper_wave, templogg):
    # Find directory
    ogrid_dir = glob.glob(path_CMFGEN)
    files_of_interest = ['obs_fin_10', 'obs_cont']    
    for d in ogrid_dir:
        dir_name = os.path.join(d, 'obs')
        files_in_dir = os.listdir(dir_name)
        #print('Now in %s' % dir_name)
        for file_in_use in files_of_interest:
            #print 'this is file_in_use: %s' % file_in_use
            if file_in_use in files_in_dir:
                #print('Going into directory: %s' % (dir_name))
                file_name = os.path.join(dir_name, file_in_use)
                print('Loading %s' % (file_name))
                # Converting file into three columns: frequencies, janskies, error
                science.claus.clausfile(file_name, dest=os.path.join(path_results))
                # Converting into Angstroms and cgs to create a text file 
                new_name = os.path.join(path_results, os.path.basename(d)+"_"+os.path.basename(file_in_use)+"_1d")
                #print 'This is new_name', new_name
                old_name = os.path.join(path_results, os.path.basename(file_in_use)+"_1d")
                #print 'This is old_name', old_name
                sp = science.spectrum.Spectrum(old_name)
                shutil.move(old_name, new_name)
                output = (sp.A, sp.cgs)
                science.spectrum.write_1d(os.path.join(os.path.abspath(path_results), new_name+"_Acgs.txt"), output)
                os.unlink(new_name)
                print ('*_Acgs.txt files have been created in the directory: %s' % (path_results))
            else:
                print('File %s is NOT in this directory: %s' % (file_in_use, dir_name))

def oldinto2cols(path_CMFGEN, path_results, lower_wave, upper_wave, templogg):
    # Find directory
    ogrid_dir = glob.glob(path_CMFGEN)
    print('Now in %s' % ogrid_dir)
    file_in_use = 'OBSFLUX'
    for dir_name in ogrid_dir:
        print'Going into directory: %s' % (dir_name)
        files_in_dir = os.listdir(dir_name)
        if file_in_use in files_in_dir:
            file_name = os.path.join(dir_name, file_in_use)
            print('loading %s' % (file_name))
            # Converting file into three columns: frequencies, janskies, error
            clausfile(file_name, dest=os.path.join(path_results))
            # Converting into Angstroms and cgs to create a text file 
            new_name = os.path.join(path_results, os.path.basename(dir_name)+"_"+os.path.basename(file_in_use)+"_1d")
            old_name = os.path.join(path_results, os.path.basename(file_in_use)+"_1d")
            sp = spectrum.Spectrum(old_name)
            shutil.move(old_name, new_name)
            #sp.save2(file_in_use+"_Acgs.txt")
            output = (sp.A, sp.cgs)
            spectrum.write_1d(os.path.join(os.path.abspath(path_results), new_name+"_Acgs.txt"), output)
            os.unlink(new_name)
        else:
            print('There is no file OBSFLUX in this directory: %s' % (dir_name))
            
def find_linear_continuum(wavflux_arr, temp_obj, plane_parallel=False):
    '''
    This function defines a continuum to CMFGEN files with a linear equation: y = m*(x -x1) + y1.
    x1, x2 = fist choice of points to create the line
    x3, x4 = second choice of points to create the line
    x5, x6 = third choice of points to make the line
    wavflux_arr = is the 2-D array of wavelength and flux
    temp_obj = is the temperature of the object. This is important because the behavior of the slope is different with temperature.
    
    ***RETURNS the array of wavelength and continuum fluxes.
    '''
    # For Teff lower than 35,000
    x1 = 1274.63 #1270.3
    x2 = 1284.5 #1290.42
    # For Teff between 35,000 and 48,000
    x3 = 1191.8 #1271.61
    x4 = 1289.8 #1298.58
    # For Teff higher than 48,000
    x5 = 1202.95 
    x6 = 1298.58

    # Start assuming that Teff is lower than 35,000
    xa = x1
    xb = x2
    # Find the closest points to x1 and x2 in the wavelength array
    temp_xa, _ = spectrum.find_nearest(wavflux_arr[0], xa)
    temp_xb, _ = spectrum.find_nearest(wavflux_arr[0], xb)
    # now find the corresponding fluxes
    ya = spectrum.findXinY(wavflux_arr[1], wavflux_arr[0], temp_xa)
    yb = spectrum.findXinY(wavflux_arr[1], wavflux_arr[0], temp_xb)
    
    if plane_parallel == True:
        xa = 1244 
        xb = 1283.6
        temp_x1, _ = spectrum.find_nearest(wavflux_arr[0], xa)
        temp_x2, _ = spectrum.find_nearest(wavflux_arr[0], xb)
        #print('closest wavelengths: %f, %f' % (temp_x1, temp_x2))
        ya_temp = spectrum.findXinY(wavflux_arr[1], wavflux_arr[0], temp_x1)
        yb_temp = spectrum.findXinY(wavflux_arr[1], wavflux_arr[0], temp_x2)
        percentage = 0.027
        ya = ya_temp + ya_temp*percentage
        yb = yb_temp + yb_temp*percentage

    # Determine which pair of x points to use according to the spectrum features, that change with temperature
    if (numpy.fabs(yb) > numpy.fabs(ya)) or (float(temp_obj) > 35000.0) and (float(temp_obj) < 48000.0):
        #print('(numpy.fabs(yb) > numpy.fabs(ya)) or (float(temp_obj) > 35000.0) and (float(temp_obj) < 48000.0)')
        xa = x3
        xb = x4
        temp_xa, _ = spectrum.find_nearest(wavflux_arr[0], xa)
        temp_xb, _ = spectrum.find_nearest(wavflux_arr[0], xb)
        ya = spectrum.findXinY(wavflux_arr[1], wavflux_arr[0], temp_xa)
        yb = spectrum.findXinY(wavflux_arr[1], wavflux_arr[0], temp_xb)
    if (float(temp_obj) > 48000.0):
        xa = x5
        xb = x6
        temp_xa, _ = spectrum.find_nearest(wavflux_arr[0], xa)
        temp_xb, _ = spectrum.find_nearest(wavflux_arr[0], xb)
        ya = spectrum.findXinY(wavflux_arr[1], wavflux_arr[0], temp_xa)
        yb = spectrum.findXinY(wavflux_arr[1], wavflux_arr[0], temp_xb)
        
    # Determine the slope of the line
    m = (yb - ya) / (xb - xa)
    # Calculate the line
    y_list = []
    for x in wavflux_arr[0]:
        y = m * (x - xa) + ya
        y_list.append(y)
    continuum_array = numpy.array([wavflux_arr[0], y_list])
    return continuum_array


