import time

import librosa
import numpy as np

from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.dispatcher import Dispatcher

from butterfly_bridge.clustering.clustering import Clustering
from butterfly_bridge.streaming.jxf import read_jxf

from butterfly_bridge.clustering.features import top_feature_indices

ip = '127.0.0.1'
port_recv = 9123
port_send = 8600
client = SimpleUDPClient(ip, port_send)

n_fft = 4 * 1024
fft_hop = 2 * 1024
sample_rate = 48000
frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)


# def get_features():
#     waveform = read_jxf('./data/waveform.jxf')[0]
#     features = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=fft_hop).T)
#     return features
# feature_shape = get_features().shape

waveform = np.random.rand(8192)
features = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=fft_hop).T)
feature_shape = features.shape

clustering = Clustering(
    epsilon=1000000,
    lambd=0.00001,
    beta=0.6,
    mu=2,
    min_samples=1
)


def message_handler(address, scope, *args):
    print(f"Received message on '{address}': {args}")

    client, clustering = scope
    # features = get_features().reshape(-1)

    waveform = np.array(args, dtype=float)
    features = np.abs(librosa.stft(y=waveform, n_fft=n_fft, hop_length=fft_hop).T)

    mean_features = clustering.append_transform(features, verbose=True)
    mean_features = np.sum(mean_features.reshape(feature_shape), axis=0)
    top_frequencies = frequencies[top_feature_indices(mean_features, n_top_features=10)[0]]

    print(f"Sending message on /bridge: {top_frequencies}")
    client.send_message('/bridge', top_frequencies)


dispatcher = Dispatcher()
dispatcher.map("/butterfly", message_handler, client, clustering)

server = BlockingOSCUDPServer((ip, port_recv), dispatcher)
server.serve_forever()
