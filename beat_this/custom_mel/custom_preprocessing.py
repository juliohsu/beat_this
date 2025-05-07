import numpy as np
import torch
import torchaudio
import librosa
from nnAudio.features import VQT, CQT

def load_audio(path, dtype="float64"):
    try:
        waveform, samplerate = torchaudio.load(path, channels_first=False)
        waveform = np.asanyarray(waveform.squeeze().numpy(), dtype=dtype)
        return waveform, samplerate
    except Exception:
        # in case torchaudio fails, try soundfile
        try:
            import soundfile as sf

            return sf.read(path, dtype=dtype)
        except Exception:
            raise RuntimeError(f'Could not load audio from "{path}".')
            # some files are not readable by soundfile, try madmom
            # try:
            #     import madmom

            #     return madmom.io.load_audio_file(str(path), dtype=dtype)
            # except Exception:
            #     raise RuntimeError(f'Could not load audio from "{path}".')

class LogMelSpectTorch(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        f_min=30,
        f_max=11000,
        n_mels=128,
        mel_scale="slaney",
        normalized="frame_length",
        power=1,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.spect_class = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            mel_scale=mel_scale,
            normalized=normalized,
            power=power,
        ).to(device)
        self.log_multiplier = log_multiplier

    def forward(self, x):
        """Input is a waveform as a monodimensional array of shape T,
        output is a 2D log mel spectrogram of shape (F,128)."""
        return torch.log1p(self.log_multiplier * self.spect_class(x).T)

# librosa
class CQTNotesSpectLibrosa(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        f_min=30.0,
        power=1,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.f_min = f_min
        self.power = power
        self.log_multiplier = log_multiplier
        self.device = device

    def forward(self, x):
        """
        Input: 1D waveform tensor (T,)
        Output: 2D log CQT spectrogram (T, 84)
        """
        # Convert tensor to numpy
        x_np = x.detach().cpu().numpy()

        # Compute complex CQT
        cqt = librosa.cqt(
            x_np,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.f_min,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

        # Get magnitude
        cqt_mag = np.abs(cqt) ** self.power

        # Convert to torch tensor
        cqt_tensor = torch.from_numpy(cqt_mag).float().to(self.device)

        # Apply log1p scaling
        return torch.log1p(self.log_multiplier * cqt_tensor).T

# nnAudio
class CQTNotesSpectNN(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        hop_length=512,
        n_bins=128,
        bins_per_octave=24,
        f_min=30.0,
        power=1,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.log_multiplier = log_multiplier
        self.cqt = CQT(sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave).to(device)

    def forward(self, x):
        """
        Input: 1D waveform tensor (T,)
        Output: 2D log CQT spectrogram (T, 128)
        """
        # Compute complex CQT magnitude
        cqt = self.cqt(x).squeeze(0) # (1, F, T) -> (F, T)

        # Apply log1p scaling
        return torch.log1p(self.log_multiplier * cqt.T)

# LIBROSA
class VQTNotesSpectLibrosa(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        hop_length=512,
        n_bins=128,
        bins_per_octave=24,
        f_min=30.0,
        gamma=0,
        power=1,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.f_min = f_min
        self.gamma = gamma  # <-- new for VQT
        self.power = power
        self.log_multiplier = log_multiplier
        self.device = device

    def forward(self, x):
        """
        Input: 1D waveform tensor (T,)
        Output: 2D log VQT spectrogram (T, 128)
        """
        # Convert tensor to numpy
        x_np = x.detach().cpu().numpy()

        # Compute complex VQT
        vqt = librosa.vqt(
            x_np,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            fmin=self.f_min,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
            gamma=self.gamma,
        )

        # Get magnitude
        vqt_mag = np.abs(vqt) ** self.power

        # Convert to torch tensor
        vqt_tensor = torch.from_numpy(vqt_mag).float().to(self.device)

        # Apply log1p scaling
        return torch.log1p(self.log_multiplier * vqt_tensor.T)

# CUSTOM
class VQTNotesSpectCustom(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        f_min=30,
        f_max=10000,
        n_bins=128,
        gamma=None,
        bins_per_octave=12,
        window_fn=torch.hann_window,
        power=1,
        normalized=True,
        log_multiplier=1000,
        device='cpu'
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.power = power
        self.normalized = normalized
        self.log_multiplier = log_multiplier
        self.device = device
        
        if gamma == None:
            self.gamma = 25.0 # positive value for vqt, 0 for cqt
        else:
            self.gamma = gamma

        self.window_fn = window_fn
        self._initialize_kernels() # initialize vqt kernel and frequencies

    def _initialize_kernels(self):
        self.frequencies = self._get_frequencies()
        self.vqt_kernels = self._get_vqt_kernels()
    
    def _get_frequencies(self): # center of frequencies for each octave bin
        factor = 2 * (1/ self.bins_per_octave)
        frequencies = self.f_min * factor ** torch.arange(0, self.n_bins, device=self.device)
        frequencies = frequencies[frequencies <= self.f_max]
        self.n_bins = len(frequencies)
        return frequencies
    
    def _get_quality_factors(self): # quality factors for each octave bin
        q_zero = 1 / (2 ** (1 / self.bins_per_octave) - 1)
        bin_indices = torch.arange(self.n_bins, device=self.device)
        if self.gamma > 0:
            q_factors = q_zero / (1 + self.gamma * (bin_indices / self.n_bins))
        else:
            q_factors = torch.ones_like(bin_indices) * q_zero
        return q_factors

    def _get_vqt_kernels(self):
        q_factors = self._get_quality_factors()
        kernels = torch.zeros(self.n_bins, self.n_fft // 2 + 1, dtype=torch.complex64, device=self.device)
        for k, (freq, q) in enumerate(zip(self.frequencies, q_factors)):
            bandwidth = freq / q
            fft_bin = freq * self.n_fft / self.sample_rate
            bw_bins = bandwidth * self.n_fft / self.sample_rate
            bin_start = max(0, int(fft_bin-bw_bins*2))
            bin_end = min(self.n_fft // 2 + 1, int(fft_bin+bw_bins*2))
            fft_indices = torch.arange(bin_start, bin_end, device=self.device)
            fft_freqs = fft_indices * self.sample_rate / self.n_fft
            sigma = bw_bins / 4
            response = torch.exp(-0.5 * ((fft_indices - fft_bin) / sigma) ** 2)
            if torch.sum(response) > 0:
                response = response / torch.sum(response)
            phase = -2 * np.pi * fft_indices * (0.5 * self.n_fft) / self.n_fft
            kernels[k, bin_start:bin_end] = response * torch.exp(1j * phase)
        return kernels
    def forward(self, x):
        x_dim = x.dim()
        if x_dim == 1:
            x = x.unsqueeze(0)
        # print(x.shape)
        window = self.window_fn(self.n_fft)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True
        )
        kernels = self.vqt_kernels.to(x.device)
        vqt = torch.matmul(kernels.unsqueeze(0).unsqueeze(0), stft) # (batch, channels, n_fft//2+1, time)
        vqt_mag = torch.abs(vqt) ** self.power
        if self.normalized:
            vqt_mag = vqt_mag / self.n_fft
        vqt_log = torch.log1p(self.log_multiplier * vqt_mag)
        if x_dim == 1:
            return vqt_log.squeeze(0).squeeze(0).transpose(0, 1) # (n_bins, time)
        return vqt_log.transpose(2, 3).squeeze(0).squeeze(0) # (batch, channels, time, n_bins) -> (n_bins, time)
    
# nnAudio
class VQTNotesSpectNN(torch.nn.Module):
    def __init__(
        self,
        sample_rate=22050,
        hop_length=512,
        n_bins=128,
        bins_per_octave=24,
        f_min=30.0,
        gamma=0,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        self.log_multiplier = log_multiplier
        self.vqt = VQT(sr=sample_rate, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave).to(device)

    def forward(self, x):
        """
        Input: 1D waveform tensor (T,)
        Output: 2D log VQT spectrogram (T, 128)
        """
        # Compute VQT already in magnitude
        vqt = self.vqt(x).squeeze(0) # (1, F, T) -> (F, T)

        # Apply log1p scaling
        return torch.log1p(self.log_multiplier * vqt.T)