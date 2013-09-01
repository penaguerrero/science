import types
import numpy
from StringIO import StringIO
from array import array
import string
import os
import tempfile
from .. import output

class clausfile:
    def __init__(self, filename, dest=""):
        self.REPAIR_POW = 0x01
        self.name = filename
        self.data = None
        if dest == "":
            self.generate_1d(filename, filename+'_1d')
        else:
            self.generate_1d(filename, os.path.join(dest, os.path.basename(filename)+'_1d'))

        if self.data == None:
            raise Exception("no data")
        
        check_x, check_y = self.data
        if len(check_x) <= 1 or len(check_y) <= 1:
            self.name = self.repair(self.REPAIR_POW)

        
    def set_data(self, data):
        self.data = data
    
    def get_data(self):
        return self.data
        
    def get_dimensions(self):
        return self.data.shape
    
    def generate_1d(self, filename, outfile):
        str_x = ""
        str_y = ""
        col_x = []
        col_y = []
        in_continuum = False
        in_intensity = False
        in_data = False
        count = 0
        try:
            fp = open(filename, 'r')
        except IOError as e:
            print(e.strerror)
            return None
        self.fp = fp
        for line in fp.readlines():
            line = string.lstrip(line, ' ')
            line = string.rstrip(line, '\n')
            if line.startswith('Continuum'):
                in_continuum = True
                print('Continuum header [%d]: "%s"' % (count, line) )
                continue
            elif line.startswith('Observed'):
                in_continuum = False
                in_intensity = True
                print('Intensity header [%d]: "%s"' % (count, line))
                continue
            elif len(line) <= 1:
                in_data = False
                continue
            else:
                in_data = True
                line = line.replace('-', 'E-')
                line = line.replace('EE-', 'E-')
            
            if in_continuum == True and in_data == True:
                str_x += " " + line
            
            if in_intensity == True and in_data == True:
                str_y += " " + line
            count += 1
                
        str_x = string.replace(str_x, 'E', 'e')
        str_y = string.replace(str_y, 'E', 'e')
        for x in string.split(str_x):
            col_x.append(x)
        for y in string.split(str_y):
            col_y.append(y)
        self.set_data(numpy.array([col_x, col_y]))
        output.ascii.dump(outfile, col_x, col_y)
    
 
    def repair(self, mask):
        print("Repairing %s" % (self.name))
        if mask == (mask & self.REPAIR_POW):
            fp = self.fp
            try:
                fp.seek(0, os.SEEK_SET)
                fp_temp = open(self.fp.name + "_repaired", 'w+')
            except IOError as e:
                print("%s: %s" % (self.name, e.strerror))
                return -1
            
            for line in fp.readlines():
                line = line.replace('-', 'E-')
                line = line.replace('EE-', 'E-')
                fp_temp.write(line)
            
            self.fp = fp_temp
            name = fp_temp.name
            fp.close()
            return name
        else:
            print("No repair mask defined")

