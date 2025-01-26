import matplotlib.pyplot as plt
import numpy as np
import itertools

files = ['/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/ExoCLASS_perso/DarkAgesModule/output_DarkAges_dist.tmp.dat', '/Users/vpoulin/Dropbox/Labo/ProgrammeCMB/ExoCLASS_perso/output/DM_decay_60_sd_distortions.dat']
data = []
for data_file in files:
    data.append(np.loadtxt(data_file))
roots = ['output_DarkAges_dist', 'DM_decay_60_sd_distortions']

fig, ax = plt.subplots()

index, curve = 0, data[0]
y_axis = ['SD_tot']
tex_names = ['SD_tot']
x_axis = 'Frequency nu [GHz]'
ylim = []
xlim = [10.0, 2000.0]
ax.loglog(curve[:, 0], abs(curve[:, 1]))

index, curve = 1, data[1]
y_axis = ['SD_tot']
tex_names = ['SD_tot']
x_axis = 'Frequency nu [GHz]'
ylim = []
xlim = [10.0, 2000.0]
ax.loglog(curve[:, 1], abs(curve[:, 2]))

ax.legend([root+': '+elem for (root, elem) in
    itertools.product(roots, y_axis)], loc='best')

ax.set_xlabel('Frequency nu [GHz]', fontsize=16)
ax.set_xlim(xlim)
ax.set_ylim()
plt.show()