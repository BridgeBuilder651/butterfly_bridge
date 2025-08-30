
import numpy as np

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def top_feature_indices(features, n_top_features: int = 10, sigma=None):
    if sigma is not None:
        features = gaussian_filter1d(features, sigma=sigma)

    peaks, properties = find_peaks(features, height=(None, None))
    heights = properties['peak_heights']

    sorted_indices = np.argsort(heights)[-n_top_features:]

    peaks = peaks[sorted_indices]

    if len(peaks) < n_top_features:
        all_indices = np.argsort(mean_feature)[::-1]
        for i in all_indices:
            if i not in peaks:
                peaks = np.append(peaks, i)
                if len(peaks) >= n_top_features:
                    break
    if len(peaks) < n_top_features:
        peaks = np.concatenate([peaks, [peaks[0]] * (n_top_features - len(peaks))])

    return peaks, features


# waveform = read_jxf('./data/waveform.jxf')[0]
# features = get_features()
# feature_mean = np.sum(get_features(), axis=0)
# top_indices, filtered = top_feature_indices(feature_mean, sigma=2.0)
#
# plt.figure(1); plt.clf()
# plt.subplot(2, 1, 1)
# plt.plot(waveform)
# plt.subplot(2, 1, 2)
# for f in features:
#     plt.plot(frequencies, f)
# plt.plot(frequencies, feature_mean, color='darkblue')
# plt.plot(frequencies, filtered, color='darkred')
# plt.scatter(frequencies[top_indices], filtered[top_indices], s=50, color='red')