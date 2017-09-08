
#import sys
#print sys.version

import matlab.engine
import dpp  # must come after import matlab.engine...
import numpy.random


mat = numpy.random.randint(0, 10, (4, 4))
print mat

S = dpp.sample_dpp(mat, 1)
print S

S = dpp.sample_dpp(mat, 2)
print S


