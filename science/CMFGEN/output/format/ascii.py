def dump(filename, x, y):
    """Generate a two column ascii file consisting of 'x' and 'y'"""
    try:
        fp = open(filename, 'w')
        for i in range(len(x)):
            fp.write(str(x[i]) + ' ' + str(y[i]) + ' 0.000' + '\n')
        fp.close()
    except IOError as e:
        print("%s: %s" % (filename, e.strerror))
        return False