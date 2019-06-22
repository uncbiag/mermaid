from __future__ import print_function
from __future__ import absolute_import

from builtins import str
from builtins import range

import os
import shutil
import glob
import re

import matplotlib.pyplot as plt
import numpy as np

def read_energies_from_log_file(log_file_name):
    # Energy format
    # Epoch 00000: Current energies: E=[37.10616], simE=[37.10353], regE=[0.00263], optE=[0.00000]

    E = []
    simE = []
    regE = []
    optE = []

    last_read_epoch = None

    with open(log_file_name, 'r') as file_object:
        line = file_object.readline()
        while line:

            if 'Epoch' in line:
                # parse it

                epoch_str = re.findall('Epoch.[0-9]*:', line)
                current_epoch = int(epoch_str[0][6:-1])
                if current_epoch != last_read_epoch:
                    last_read_epoch = current_epoch

                    energy_values = re.findall('\[.*?\]', line)

                    assert (len(energy_values) == 4)

                    E.append(float(energy_values[0][1:-1]))
                    simE.append(float(energy_values[1][1:-1]))
                    regE.append(float(energy_values[2][1:-1]))
                    optE.append(float(energy_values[3][1:-1]))

            line = file_object.readline()

    file_object.close()

    return E,simE,regE,optE

stage_nrs = [0,1,2]
root_dir = 'logs'

E = []
simE = []
regE = []
optE = []

lens = []

for stage_nr in stage_nrs:
    log_file_name = os.path.join(root_dir,'log_stage_{}_opt.txt'.format(stage_nr))
    print('Reading: {}'.format(log_file_name))
    c_E,c_simE,c_regE,c_optE = read_energies_from_log_file(log_file_name)

    lens.append(len(c_E))

    E += c_E
    simE += c_simE
    regE += c_regE
    optE += c_optE

cum_lens = np.array(lens).cumsum()

plt.plot(E)
plt.title('Energy')
for l in cum_lens:
    plt.axvline(x=l,color='r',linestyle=':')
plt.show()
