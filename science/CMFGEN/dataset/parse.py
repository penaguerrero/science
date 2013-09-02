import numpy
import string
import os
from science.CMFGEN import output

class CMFGENException(Exception):
    pass

class LoadFile:
    def __init__(self, filename, dest="", *kwargs):
        self.name = filename
        self.data = None
        self.verbose = False

        self.check_syntax()

        if not dest:
            self.generate_1d(self.name, os.path.splitext(self.name)[0] + '.1d')
        else:
            self.generate_1d(self.name, os.path.join(dest, os.path.splitext(os.path.basename(self.name))[0] + '.1d'))

        if self.data == None:
            raise CMFGENException("{} contains no data".format(self.name))

        if 'verbose' in kwargs:
            self.verbose = kwargs['verbose']

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def get_dimensions(self):
        return self.data.shape

    def check_syntax(self):
        found = 0
        headers = ['Continuum', 'Observed']
        for header in headers:
            for line in open(self.name, 'r').readlines():
                line = string.lstrip(line, ' ')
                line = string.rstrip(line, '\n')
                if line.startswith(header):
                    found += 1
        if found < 1:
            raise CMFGENException("{} file structure is invalid".format(self.name))

        return True

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
                if self.verbose:
                    print('Continuum header [%d]: "%s"' % (count, line))
                continue
            elif line.startswith('Observed'):
                in_continuum = False
                in_intensity = True
                if self.verbose:
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
