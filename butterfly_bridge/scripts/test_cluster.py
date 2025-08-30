import numpy as np
import tqdm
from scipy.io import wavfile

from scipy.signal import ShortTimeFFT
from scipy.signal import find_peaks

from butterfly_bridge.visualization.plotting import plt, pg, pv, pvq

from butterfly_bridge.streaming.spectrogram import Spectrogram

from umap import UMAP

# plot
verbose = True

# Path to your WAV file
file_name = './data/footsteps.wav'

# Read the WAV file
sample_rate, audio = wavfile.read(file_name)

print(f"Sampling rate: {sample_rate} Hz")
print(f"Data shape: {audio.shape}")
print(f"Data type: {audio.dtype}")
print(f"Data duration: {audio.shape[0] / sample_rate}[s]")

if verbose:
    pg.plot(audio[0:60 * sample_rate])
    pg.plot(audio[60 * sample_rate: 70 * sample_rate])

ids = np.array([0, 17, 60, 83, 106])
names = ['moses', 'jacob', 'nico', 'kate', 'artemis']


#%%


sample = audio[237500:250000]
pg.plot(sample)


#%% audio feature extraction

import librosa

n_mfcc = 128
y = np.asarray(sample, dtype=float)
mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, hop_length=256).T

#%%

plt.imshow(mfccs[:, 2:].T, aspect='auto', origin='lower', clim=(0, 20))

#%%

pg.image(mfccs)


#%%
from butterfly_bridge.visualization.plotting import pg


img = mfccs
img /= (np.max(img, axis=0) + 1e-10)

img[:, 0] = 0

win = pg.GraphicsLayoutWidget(show=True, title="Features")
view = win.addViewBox()
# view.setAspectLocked(True)  # Keep aspect ratio fixed for image

im = pg.ImageItem(img)
view.addItem(im)
view.setXRange(0, len(audio))

norm = mfccs.shape[1] / 2

x1 = np.linspace(0, len(mfccs) - 1, len(y))
y1 = y / np.max(np.abs(y)) * norm

curve = pg.PlotCurveItem(x=x1, y=y1, pen='orange')
view.addItem(curve)

x2 = np.linspace(0, len(mfccs) - 1, len(img))
#y2 = np.sum(img, axis=1)\
#y2 = np.max(img, axis=1) - np.min(img, axis=1)
y2 = np.sum(np.sign(img) * img, axis=1)
y2 = y2 * y2
y2 = y2 / y2.max() * norm

curve = pg.PlotCurveItem(x=x2, y=y2, pen='red')
view.addItem(curve)

view.show()
#%%

plt.plot(y2)
plt.show()


#%%

chunk_size = 1024 * 10
size = chunk_size * 100

s = Spectrogram(size=size, chunk_size=chunk_size, sample_rate=48000)
s.update(audio)


#%%
from butterfly_bridge.visualization.plotting import pg
pg.image(s.to_array())


#%%

from librosa.feature import melspectrogram

ms = melspectrogram(y=np.asarray(audio, dtype=float)).T
pg.image(ms)



#%%

total_power = np.sum(power, axis=1)
p = pg.plot(total_power)

peaks = find_peaks(total_power, height=100)

bins = np.asarray(ids * spectrum_sample_rate, dtype=int)
bins = np.concatenate([bins, [len(audio)]])

person = np.digitize(peaks[0], bins=bins)

plt.scatter(np.arange(len(peaks[0])), peaks[0], c=person)
plt.show()

#%%
i = peaks[0][50]
d = 50
pg.image(np.abs(spectrum[:, i-d: i+d]).T)

#%%

d = 2
data = np.array([np.sum(np.abs(spectrum[p-d:p+d]), axis=0) for p in peaks[0]])
data = np.clip(data, None, 500)

#%%
pi = pg.image(data)

#%%



#%%
from umap import UMAP
umap = UMAP(n_components=3, n_neighbors=10)  # , min_dist=0.5, spread=1)
y = umap.fit_transform(data)

if y.shape[1] == 2:
    y = np.concatenate([y, np.zeros((len(y), 1))], axis=1)
pl = pvq.BackgroundPlotter()
pl.add_mesh(y, point_size=15, scalars=person, cmap='rainbow')


#%% data

file_name_text = file_name[:-3] + 'txt'
footsteps = np.asarray(np.loadtxt(file_name_text, delimiter=','), dtype=int)
footsteps_indices = np.asarray(footsteps[:, 1] * sample_rate / 1000, dtype=int)

window_left = int(0.5 * sample_rate)
window_right = int(0.1 * sample_rate)

data = [audio[s-window_left:s+window_right] for s in footsteps_indices]
data = np.array([d for d in data if len(d) == window_right + window_left], dtype=float)
data /= np.percentile(np.abs(data), 99)
data = np.clip(data, -1, 1)

if verbose:
    p = pg.image(data.T)
    p.view.setAspectLocked(False)
    p.setColorMap(pg.colormap.get('magma'))

#%% audio feature extraction

import librosa

n_mfcc = 128
y = np.asarray(audio[:sample_rate * 100], dtype=float)
mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, hop_length=1024)


pg.image(mfccs.T)

#%% format text file

def format_text_file(file_name: str, file_name_formatted: str):
    with open(file_name_text) as file:
        text = file.read()
    text = text.replace(';', '')
    with open('./data/footsteps_new.txt', 'w') as file:
        file.write(text)

# format_text_file(file_name_text, file_name_text)
