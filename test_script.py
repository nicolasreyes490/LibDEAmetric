# DEA metric evaluator import
from base_models import *

path_source_file = "dataepsilon"        # path to source file with DMU evaluate
m = 3                                   # inputs to minimize
s = 0                                   # outputs yo maximize
DMU = 30                                # number of DMU

performance_metric(path_source_file, m, s, DMU)        # call for performance metric
