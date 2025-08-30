import os
import glob
import numpy as np

data_dir = './data/acoustic_footstep/AFPID-II_P2_FE1'
data_types = ['handcraft', 'spectrogram', 'waveform']

data_files = {}
for data_type in data_types:
    data_files[data_type] = sorted(glob.glob(os.path.join(data_dir, data_type, '*.npy')))
data_files['subject'] = np.array([int(f.split('_')[-4][1:]) for f in data_files['waveform']])


#%%

i = 90
waveform = np.load(data_files['waveform'][i])
spectrum = np.load(data_files['spectrogram'][i])
handcraft = np.load(data_files['handcraft'][i])

plt.figure(1)
plt.clf()
plt.subplot(1, 3, 1)
plt.plot(waveform)
plt.subplot(1, 3, 2)
plt.imshow(spectrum, origin='lower')
plt.subplot(1, 3, 3)
plt.imshow(handcraft, origin='lower')

#%%

from butterfly_bridge.visualization.plotting import pg, plt

plt.figure(1)
plt.clf()
k = 0
for i in range(25):
    plt.subplot(5, 5, i + 1)
    for k in range(1):
        plt.plot(np.load(data_files['waveform'][5 * i + k]))


#%%

import umap

n = 5000
di = 10
X = np.array([np.load(data_files['handcraft'][i*di]).reshape(-1) for i in range(n)])
Y = umap.UMAP(n_components=3, n_neighbors=15).fit_transform(X)

#%%
from butterfly_bridge.visualization.plotting import pvq

points = pv.PolyData(Y)
points['colors'] = data_files['subject'][[i * di for i in range(n)]]

plotter = pvq.BackgroundPlotter()
plotter.add_points(points, scalars='colors', pickable=True)

fig, ax = plt.subplots(tight_layout=True)
line, = ax.plot([], [])
waveform_chart = pv.ChartMPL(fig, size=(0.4, 0.3), loc=(0.58, 0.05))
waveform_chart.background_color = (1.0, 1.0, 1.0, 0.5)  # Semi-transparent white
plotter.add_chart(waveform_chart)

def update_waveform_inset(picked_point):
    index = points.find_closest_point(picked_point)
    waveform = np.load(data_files['waveform'][index])
    subject = points['colors'][index]

    ax.clear()
    ax.plot(waveform, color=plt.colormaps.get('viridis')(subject/40.))
    ax.set_title(f"Waveform {index} - {subject}")
    waveform_chart.render()

plotter.enable_point_picking(callback=update_waveform_inset, picker='point', left_clicking=True)

#%%



