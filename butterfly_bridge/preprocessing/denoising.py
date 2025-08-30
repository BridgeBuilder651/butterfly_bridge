# -*- coding: utf-8 -*-
"""
Butterfly Bridge

Online denoising algorithm for footstep analysis

Author: Christoph Kirst
Email: christoph.kirst.ck@gmail.com
Copyright 2025 Christoph Kirst

Examples
--------
>>> import numpy as np
>>> import librosa
>>> from butterfly_bridge.preprocessing.denoising import Denoising
>>> sr, duration = 22050, 1
>>> t = np.linspace(0, duration, int(sr * duration), endpoint=False)
>>> clean_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
>>> noise_profile = librosa.util.normalize(librosa.load(librosa.example('trumpet'))[0][:int(duration * sr)])
>>> noisy_audio = clean_audio + noise_profile
>>> denoised_audio = Denoising(noise, n_fft=2048, hop_length=512, alpha=4.0)(noisy_audio)
>>> n = len(denoised_audio)

>>> from butterfly_bridge.visualization.plotting import plt
>>> plt.figure(1, figsize=(12, 8)); plt.clf()
>>> plt.plot(noisy_audio)
>>> plt.title('Noisy Audio')
>>> plt.subplot(4, 1, 2)
>>> plt.plot(noise_profile)
>>> plt.title('Noise Profile')
>>> plt.ylim(-1, 1)
>>> plt.subplot(4, 1, 3)
>>> plt.plot(denoised_audio)
>>> plt.title('Denoised Audio')
>>> plt.ylim(-1, 1)
>>> plt.subplot(4, 1, 4)
>>> plt.plot(clean_audio)
>>> plt.title('Original Audio')
>>> plt.ylim(-1, 1)
>>> plt.tight_layout()
>>> plt.show()
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


class Denoising:
    def __init__(self, noise_profile,  n_fft: int, hop_length: int, alpha: float = 1.0, beta: float = 0.005):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.alpha = alpha
        self.beta = beta

        self.noise_power = None
        self.initialize_noise_profile(noise_profile)

    def initialize_noise_profile(self, noise_profile):
        noise_stft = librosa.stft(noise_profile, n_fft=self.n_fft, hop_length=self.hop_length)
        noise_magnitude = np.abs(noise_stft)
        self.noise_power = np.mean(noise_magnitude ** 2, axis=1, keepdims=True)

    def denoise(self, noisy_audio):
        noisy_stft = librosa.stft(noisy_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        noisy_magnitude, noisy_phase = librosa.magphase(noisy_stft)
        noisy_power = noisy_magnitude ** 2

        subtracted_power = noisy_power - self.alpha * self.noise_power
        cleaned_power = np.maximum(subtracted_power, self.beta * self.noise_power)
        cleaned_stft = np.sqrt(cleaned_power) * noisy_phase
        return librosa.istft(cleaned_stft, hop_length=self.hop_length)

    def __call__(self, noisy_audio):
        return self.denoise(noisy_audio)
