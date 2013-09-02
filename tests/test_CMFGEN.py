import os
import unittest
from science.CMFGEN import dataset
from science import utility
DATA = utility.DataDir('data/CMFGEN')
RESULT = utility.DataDir('result/CMFGEN', create=True)
CONTINUUM_DATA = os.path.join(DATA.path, 'testfile_cont.txt')
FIN_DATA = os.path.join(DATA.path, 'testfile_fin.txt')
BAD_DATA = os.path.join(DATA.path, 'testfile_bad.txt')


class LoadFileOperation(unittest.TestCase):
    def setUp(self):
        self.continuum = dataset.LoadFile(CONTINUUM_DATA, RESULT.path)
        self.fin = dataset.LoadFile(FIN_DATA, RESULT.path)

    def test_continuum_dat_isOK(self):
        self.assertIsNotNone(self.continuum.get_data())

    def test_fin_data_isOK(self):
        self.assertIsNotNone(self.fin.get_data())

    def test_bad_data_causes_LoadFileException(self):
        with self.assertRaises(dataset.CMFGENException):
            dataset.LoadFile(BAD_DATA)

def suite():
    s = unittest.TestSuite()
    s.addTest(LoadFileOperation())
    return s

if __name__ == '__main__':
    unittest.main()
