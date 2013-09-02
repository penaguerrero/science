def dump(filename, x, y):
    """Generate a 'claus' file.  Two headers that separate a strangely formed 1d array"""
    try:
        fp = open(filename, 'w')
        fp.write('\n\n')
        fp.write('Continuum Frequencies\n\n')
        for wave in x:
            fp.write(str(wave) + '\n')
        fp.write('\n\n')
        fp.write('Observed intensity\n\n')
        for flux in y:
            fp.write(str(flux) + '\n')
        fp.write(' ')
        fp.close()
    except IOError as e:
        print("%s: %s" % (filename, e.strerror))
        return False
