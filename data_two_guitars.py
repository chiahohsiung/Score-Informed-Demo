from utils import load_audio, load_info, midi_to_mask
from pathlib import Path
import torch.utils.data
import argparse
import random
import musdb
import torch
import tqdm
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os 
import json


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio


def load_datasets(parser, args):
    """Loads the specified dataset from commandline arguments

    Returns:
        train_dataset, validation_dataset
    """
    if args.dataset == 'trackfolder_fix':
        parser.add_argument('--target-file', type=str)
        parser.add_argument('--interferer-files', type=str, nargs="+")
        parser.add_argument(
            '--random-track-mix',
            action='store_true', default=False,
            help='Apply random track mixing augmentation'
        )
        parser.add_argument(
            '--source-augmentations', type=str, nargs='+',
            default=['gain', 'channelswap']
        )

        args = parser.parse_args()
#         args.target = Path(args.target_file).stem
        args.target_file = 'gt_0.wav'
        args.interferer_files = ['gt_1.wav']
        dataset_kwargs = {
            'root': Path(args.root),
            'interferer_files': args.interferer_files,
            'target_file': args.target_file
        }

        source_augmentations = Compose(
            [globals()['_augment_' + aug] for aug in args.source_augmentations]
        )

        train_dataset = FixedSourcesTrackFolderDataset(
            split='train',
            source_augmentations=source_augmentations,
            random_track_mix=args.random_track_mix,
            random_chunks=True,
            seq_duration=args.seq_dur,
            **dataset_kwargs
        )

        valid_dataset = FixedSourcesTrackFolderDataset(
            split='valid',
            seq_duration=args.seq_dur,
            **dataset_kwargs
        )
        


    elif args.dataset == 'sourcefolder':
        parser.add_argument('--interferer-dirs', type=str, nargs="+")
        parser.add_argument('--target-dir', type=str)
        parser.add_argument('--ext', type=str, default='.wav')
        parser.add_argument('--nb-train-samples', type=int, default=1000)
        parser.add_argument('--nb-valid-samples', type=int, default=100)
        parser.add_argument(
            '--source-augmentations', type=str, nargs='+',
            default=['gain', 'channelswap']
        )
        args = parser.parse_args()
        args.target = args.target_dir

        dataset_kwargs = {
            'root': Path(args.root),
            'interferer_dirs': args.interferer_dirs,
            'target_dir': args.target_dir,
            'ext': args.ext
        }

        source_augmentations = Compose(
            [globals()['_augment_' + aug] for aug in args.source_augmentations]
        )

        
        train_dataset = SourceFolderDataset(
            split='train',
            source_augmentations=source_augmentations,
            random_chunks=True,
            nb_samples=args.nb_train_samples,
            seq_duration=args.seq_dur, 
            **dataset_kwargs
        )

        valid_dataset = SourceFolderDataset(
            split='test',
            random_chunks=True,
            seq_duration=args.seq_dur,
            nb_samples=args.nb_valid_samples,
            **dataset_kwargs
        )


    return train_dataset, valid_dataset, args


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
        

class FixedSourcesTrackFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split='train',
        target_file='gt_0.wav',
        interferer_files=['gt_1.wav'],
        seq_duration=None,
        random_chunks=False,
        random_track_mix=False,
        source_augmentations=lambda audio: audio,
        sample_rate=44100,
    ):
        """A dataset of that assumes audio sources to be stored
        in track folder where each track has a fixed number of sources.
        For each track the users specifies the target file-name (`target_file`)
        and a list of interferences files (`interferer_files`).
        A linear mix is performed on the fly by summing the target and
        the inferers up.

        Due to the fact that all tracks comprise the exact same set
        of sources, the random track mixing augmentation technique
        can be used, where sources from different tracks are mixed
        together. Setting `random_track_mix=True` results in an
        unaligned dataset.
        When random track mixing is enabled, we define an epoch as
        when the the target source from all tracks has been seen and only once
        with whatever interfering sources has randomly been drawn.

        This dataset is recommended to be used for small/medium size
        for example like the MUSDB18 or other custom source separation
        datasets.

        Example
        =======
        train/1/vocals.wav ---------------\
        train/1/drums.wav (interferer1) ---+--> input
        train/1/bass.wav -(interferer2) --/

        train/1/vocals.wav -------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.random_track_mix = random_track_mix
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        # set the input and output files (accept glob)
        self.target_file = target_file
        self.interferer_files = interferer_files
        self.source_files = self.interferer_files + [self.target_file]
        self.tracks = list(self.get_tracks())
        self.stft = STFT()
        self.spec = Spectrogram(mono=False)


    def __getitem__(self, index):
        # first, get target track
        track_path = self.tracks[index]['path']
        min_duration = self.tracks[index]['min_duration']
        
        if self.random_chunks:
            # determine start seek by target duration
            start = random.uniform(0, min_duration - self.seq_duration)
        else:
            start = 0

        # assemble the mixture of target and interferers
        audio_sources = []
        midi_sources = []
        start_ends = []
        # load target
        # random choose target 

        self.source_files = random.sample(self.source_files, len(self.source_files))

        for index, source in enumerate(self.source_files):      

            if self.random_chunks:
                # determine start seek by target duration
                start = random.uniform(0, min_duration - self.seq_duration)
            else:
                start = 0
                        
            audio = load_audio(
                track_path / source, start=start, dur=self.seq_duration
            )
            audio = torch.unsqueeze(self.source_augmentations(audio), 0)
            print('audio.shape', audio.shape)
            audio_sources.append(audio)

            start_ends.append((start, start+self.seq_duration))
            midi_path = os.path.join(str(track_path), source.split('.')[0] + '.txt')
            midi_sources.append(midi_path)


        stems = torch.stack(audio_sources)
        # # apply linear mix over source index=0
        x = stems.sum(0)
        # target is always the first element in the list
        y = stems[-1]

        # time series to stft
        x = self.stft.forward(x)
        x = self.spec.forward(x)
        
        y = self.stft.forward(y)
        y = self.spec.forward(y)
        
        # Hard Mask to Soft Mask
        
        mask_accom = midi_to_mask(x.permute(1, 0, 2)[0].numpy(), midi_sources[0], start_ends[0])
        mask_target = midi_to_mask(x.permute(1, 0, 2)[0].numpy(), midi_sources[-1], start_ends[-1])
        mask_target = mask_target / (mask_target + mask_accom)

        x_filtered = mask_target * x.permute(1, 0, 2)[0].numpy()
        
        # Expand dimensions for the model
        x_filtered = torch.tensor(np.expand_dims(x_filtered, 1))
        
        return x, y, x_filtered 


    def __len__(self):
        return len(self.tracks)

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        good = 0
        for track_path in tqdm.tqdm(p.iterdir()):
            if good >= 2000 and self.split == 'train':
                break
            elif good >= 200 and self.split == 'valid':
                break
            if track_path.is_dir():
                source_paths = [track_path / s for s in self.source_files]

                if not all(sp.exists() for sp in source_paths):
                    continue
                good += 1
                
                if self.seq_duration is not None:
                    infos = list(map(load_info, source_paths))
                    # get minimum duration of track
                    min_duration = min(i['duration'] for i in infos)
                    if min_duration >= self.seq_duration:
                        yield({
                            'path': track_path,
                            'min_duration': min_duration
                        })
                else:
                    # path: train/song1
                    # under that exists audios and midis
                    yield({'path': track_path, 'min_duration': None})
                
        print('good', good)

class SourceFolderDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split='train',
        target_dir='vocals',
        interferer_dirs=['bass', 'drums'],
        ext='.wav',
        nb_samples=1000,
        seq_duration=None,
        random_chunks=False,
        sample_rate=44100,
        source_augmentations=lambda audio: audio,
    ):
        """A dataset of that assumes folders of sources,
        instead of track folders. This is a common
        format for speech and environmental sound datasets
        such das DCASE. For each source a variable number of
        tracks/sounds is available, therefore the dataset
        is unaligned by design.

        Example
        =======
        train/vocals/track11.wav -----------------\
        train/drums/track202.wav  (interferer1) ---+--> input
        train/bass/track007a.wav  (interferer2) --/

        train/vocals/track11.wav ---------------------> output

        """
        self.root = Path(root).expanduser()
        self.split = split
        self.sample_rate = sample_rate
        self.seq_duration = seq_duration
        self.ext = ext
        self.random_chunks = random_chunks
        self.source_augmentations = source_augmentations
        self.target_dir = target_dir
        self.interferer_dirs = interferer_dirs
        self.source_folders = self.interferer_dirs + [self.target_dir]
        self.source_tracks = self.get_tracks()
        self.nb_samples = nb_samples

        
    def __getitem__(self, index):
        # for validation, get deterministic behavior
        # by using the index as seed
        if self.split == 'test':
            random.seed(index)

        # For each source draw a random sound and mix them together
        audio_sources = []
        
        for source in self.source_folders:
            # select a random track for each source
            source_path = random.choice(self.source_tracks[source])
            if self.random_chunks:
                duration = load_info(source_path)['duration']
                start = random.uniform(0, duration - self.seq_duration)
                
            else:
                start = 0

            audio = load_audio(
                source_path, start=start, dur=self.seq_duration
            )

            audio = self.source_augmentations(audio)
            audio_sources.append(audio)
        stems = torch.stack(audio_sources)
        # apply linear mix over source index=0
        x = stems.sum(0)
        # target is always the last element in the list
        y = stems[-1]
        return x, y

    def __len__(self):
        return self.nb_samples

    def get_tracks(self):
        """Loads input and output tracks"""
        p = Path(self.root, self.split)
        source_tracks = {}
        for source_folder in tqdm.tqdm(self.source_folders):
            tracks = []
            source_path = (p / source_folder)
            for source_track_path in source_path.glob('*' + self.ext):
                if self.seq_duration is not None: 
                    info = load_info(source_track_path)
                    if info['duration'] >= self.seq_duration:
                        tracks.append(source_track_path)
                else:
                    tracks.append(source_track_path)
            source_tracks[source_folder] = tracks

        return source_tracks



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Open Unmix Trainer')
    parser.add_argument(
        '--dataset', type=str, default="musdb",
        choices=[
            'musdb', 'aligned', 'sourcefolder',
            'trackfolder_var', 'trackfolder_fix'
        ],
        help='Name of the dataset.'
    )

    parser.add_argument(
        '--root', type=str, help='root path of dataset'
    )

    parser.add_argument(
        '--save',
        action='store_true',
        help=('write out a fixed dataset of samples')
    )

    parser.add_argument('--target', type=str, default='vocals')

    # I/O Parameters
    parser.add_argument(
        '--seq-dur', type=float, default=5.0,
        help='Duration of <=0.0 will result in the full audio'
    )

    parser.add_argument('--batch-size', type=int, default=16)

    args, _ = parser.parse_known_args()
    train_dataset, valid_dataset, args = load_datasets(parser, args)

    # Iterate over training dataset
    total_training_duration = 0
    for k in tqdm.tqdm(range(len(train_dataset))):
        x, y = train_dataset[k]
        total_training_duration += x.shape[1] / train_dataset.sample_rate
        if args.save:
            import soundfile as sf
            sf.write(
                "test/" + str(k) + 'x.wav',
                x.detach().numpy().T,
                44100,
            )
            sf.write(
                "test/" + str(k) + 'y.wav',
                y.detach().numpy().T,
                44100,
            )

    print("Total training duration (h): ", total_training_duration / 3600)
    print("Number of train samples: ", len(train_dataset))
    print("Number of validation samples: ", len(valid_dataset))

    # iterate over dataloader
    train_dataset.seq_duration = args.seq_dur
    train_dataset.random_chunks = True

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    )

    for x, y in tqdm.tqdm(train_sampler):
        pass
