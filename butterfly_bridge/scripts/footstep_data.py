
import tqdm
import numpy as np
import librosa
import umap

from umap import UMAP
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT
from scipy.signal import find_peaks

from butterfly_bridge.visualization.plotting import plt, pg, pv, pvq
from butterfly_bridge.streaming.spectrogram import Spectrogram

# Path to your WAV file
file_name = './data/footsteps.wav'

# Read the WAV file
sample_rate, audio = wavfile.read(file_name)

print(f"Sampling rate: {sample_rate} Hz")
print(f"Data shape: {audio.shape}")
print(f"Data type: {audio.dtype}")
print(f"Data duration: {audio.shape[0] / sample_rate}[s]")

pg.plot(audio)
# pg.plot(audio[60 * sample_rate: 70 * sample_rate])

ids = np.array([0, 17, 60, 83, 106])
names = ['moses', 'jacob', 'nico', 'kate', 'artemis']
switch = ids * sample_rate

#%% determine peaks

from scipy.signal.windows import gaussian
duration = 4096  # int(sample_rate * 0.2)
window = gaussian(duration, std=duration / 2, sym=True)  # symmetric Gaussian window

hop = 512  # int(sample_rate * 0.01)
spectrum_sample_rate = int(sample_rate / hop)

stft = ShortTimeFFT(win=window, hop=hop, fs=sample_rate, scale_to='psd')
spectrum = stft.stft(audio).T
power = np.abs(spectrum)
total_power = np.sum(power, axis=1)

imv = pg.image(power)
imv.setColorMap(pg.colormap.get('magma'))
view = imv.getView()
view.setXRange(0, len(total_power))
view.setYRange(0, np.max(total_power))

curve = pg.PlotCurveItem(x=np.arange(len(total_power)), y=total_power, pen='orange')
view.addItem(curve)
view.show()


peaks = find_peaks(total_power, height=250)

bins = np.asarray(ids * spectrum_sample_rate, dtype=int)
bins = np.concatenate([bins, [len(audio)]])

labels = np.digitize(peaks[0], bins=bins)

plt.figure(1); plt.clf()
plt.scatter(np.arange(len(peaks[0])), peaks[0], c=labels)
plt.show()

#%% data samples around peaks

waveforms = np.array([audio[p * hop - 2024: p * hop + 8192] for p in peaks[0]], dtype=float)
# pg.image(waveforms.T)


n_mfcc = 128
mfccs = np.array([librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, hop_length=1024).T for y in waveforms])

features = mfccs.reshape(len(mfccs), -1)
pg.image(features.T)

plt.figure(2); plt.clf()
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(mfccs[i, :, 1:], origin='lower', aspect='auto')


#%%
from scipy.cluster.vq import whiten
import umap
from butterfly_bridge.visualization.plotting import plot_embedding_3d

features_whitened = whiten(features)

Y = umap.UMAP(n_components=2, n_neighbors=3).fit_transform(features_whitened)
Y = np.pad(Y, ((0, 0), (0, 3 - Y.shape[1])))

def plot_item(fig, index):
    plt.figure(fig.number)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(waveforms[index])
    plt.title(f'person={names[labels[index]-1]}')
    plt.subplot(2, 1, 2)
    plt.imshow(mfccs[index, :, 1:])

plot_embedding_3d(Y, labels, plot_item)

p = plot_embedding_3d(Y, labels)
actor =list(p.actors.values())[0]
actor.prop.point_size = 10


#%%

np.save('./data/waveforms.npy', waveforms)
np.save('./data/features.npy', features)
np.save('./data/labels.npy', labels)