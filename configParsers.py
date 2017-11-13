
from __future__ import print_function

import os
import argparse
import ConfigParser
import json

print(os.getcwd())
configParser = ConfigParser.RawConfigParser()
configParser.readfp(open(r'./config.txt'))

sec_name = 'boolType'
CUDA_ON = configParser.getboolean(sec_name, 'CUDA_ON')
USE_FLOAT16 = configParser.getboolean(sec_name, 'USE_FLOAT16')
loadSettingsFromFile = configParser.getboolean(sec_name, 'loadSettingsFromFile')
saveSettingsToFile = configParser.getboolean(sec_name, 'saveSettingsToFile')
useMap = configParser.getboolean(sec_name, 'useMap')
visualize = configParser.getboolean(sec_name, 'visualize')
smoothImages = configParser.getboolean(sec_name, 'smoothImages')
useRealImages = configParser.getboolean(sec_name, 'useRealImages')

sec_name = 'floatType'
myAdamEPS = configParser.getfloat(sec_name, 'myAdamEPS')
gaussianStd = configParser.getfloat(sec_name, 'gaussianStd')
simMeasureSigma = configParser.getfloat(sec_name, 'simMeasureSigma')

sec_name = 'intType'
dim = configParser.getint(sec_name, 'dim')
nrOfIterations = configParser.getint(sec_name, 'nrOfIterations')
img_len = configParser.getint(sec_name, 'img_len')

sec_name = 'strType'
modelName = configParser.get(sec_name, 'modelName')
smoothType = configParser.get(sec_name, 'smoothType')
optimName = configParser.get(sec_name, 'optimName')


sec_name = 'listType'
multi_scale_scale_factors = json.loads(configParser.get(sec_name,'multi_scale_scale_factors'))
multi_scale_iterations_per_scale = json.loads(configParser.get(sec_name,'multi_scale_iterations_per_scale'))

