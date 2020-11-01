from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils


class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
        

class STFT():
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = torch.hann_window(n_fft)
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """
        
        nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram():
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_channels, nb_bins)
            later in model: (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(1, 2)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        return stft_f.permute(1, 0, 2)


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


class OpenUnmix(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=True,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmix, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

        self.fc1 = Linear(
            self.nb_bins*nb_channels*2, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)

        if input_mean is not None:
            mix_mean = torch.from_numpy(
                -input_mean[0][:self.nb_bins]
            ).float()
            mix_filtered_mean = torch.from_numpy(
                -input_mean[1][:self.nb_bins]
            ).float()
        else:
            mix_mean = torch.zeros(self.nb_bins)
            mix_filtered_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            mix_scale = torch.from_numpy(
                1.0/input_scale[0][:self.nb_bins]
            ).float()
            mix_filtered_scale = torch.from_numpy(
                1.0/input_scale[1][:self.nb_bins]
            ).float()
        else:
            mix_scale = torch.ones(self.nb_bins)
            mix_filtered_scale = torch.ones(self.nb_bins)
        
        self.mix_mean = Parameter(mix_mean)
        self.mix_scale = Parameter(mix_scale)
        self.mix_filtered_mean = Parameter(mix_filtered_mean)
        self.mix_filtered_scale = Parameter(mix_filtered_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def forward(self, x, x_filtered):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)
        
        # nb_samples, nb_frames, nb_channels, nb_bins -> nb_frames, nb_samples, nb_channels, nb_bins 
        x = x.transpose(1, 0) 
        x_filtered = x_filtered.transpose(1, 0)
        
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape
        
        mix = x.detach().clone()
        
        # crop
        x = x[..., :self.nb_bins]
        x_filtered = x_filtered[..., :self.nb_bins]
        
        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.mix_mean
        x *= self.mix_scale
        
        x_filtered += self.mix_filtered_mean
        x_filtered *= self.mix_filtered_scale
        x = torch.cat([x, x_filtered], -1)
        
        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins*2))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix

        return x
