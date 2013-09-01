import os
import errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

class DataDirException(Exception):
    pass

class DataDir(object):
    def __init__(self, path, **kwargs):
        self.verbose = False
        self.path = os.path.abspath(path)
        
        if 'verbose' in kwargs:
                self.verbose = kwargs['verbose']
        
        if self.verbose:
            print("{}: {}".format(self.__class__.__name__, self.path))
        
        if 'create' in kwargs:
            if self.verbose:
                print("Does {} exist? ".format(self.path)),
                
            if os.path.exists(self.path):
                if self.verbose:
                    print("YES")
                return
            try:
                if self.verbose:
                    print("NO")
                mkdir_p(self.path)
            except OSError as e:
                raise DataDirException(e.strerror)                    
                
    def contents(self):
        return os.listdir(self.path)
            
 
if __name__ == "__main__":
    results = DataDir('/tmp/myresults/blah', create=True)
    print(results)
    print(results.path)