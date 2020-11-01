import torch
import numpy as np
import argparse
import soundfile as sf
import norbert
import json
from pathlib import Path
import scipy.signal
import resampy
import model_si
import utils_two_guitars
import warnings
import tqdm
from contextlib import redirect_stderr
import io
from utils_two_guitars import midi_to_mask
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os


def audio_to_stft(x, n_fft=4096, n_hop=1024, center=True):
    """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
    """
    
    window = torch.hann_window(n_fft)

    nb_channels, nb_timesteps = x.size()
    
    # merge nb_samples and nb_channels for multichannel stft
    x = x.reshape(nb_channels, -1)

    # compute stft with parameters as close as possible scipy settings
    stft_f = torch.stft(
        x,
        n_fft=n_fft, hop_length=n_hop,
        window=window, center=center,
        normalized=False, onesided=True,
        pad_mode='reflect'
    )

    # reshape back to channel dimension
    stft_f = stft_f.contiguous().view(
        nb_channels, n_fft // 2 + 1, -1, 2
    )
    return stft_f


def stft_to_spec(stft_f, power=1, mono=False):
    """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_channels, nb_bins)
            later in model: (nb_frames, nb_samples, nb_channels, nb_bins)
    """
    stft_f = stft_f.transpose(1, 2)
    # take the magnitude
    stft_f = stft_f.pow(2).sum(-1).pow(power / 2.0)

    # downmix in the mag domain
    if mono:
        stft_f = torch.mean(stft_f, 1, keepdim=True)

    # permute output for LSTM convenience
    return stft_f.permute(1, 0, 2)


def load_model(target, model_name='umxhq', device='cpu'):
    """
    target model path can be either <target>.pth, or <target>-sha256.pth
    (as used on torchub)
    """
    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        # model path does not exist, use hubconf model
        try:
            # disable progress bar
            err = io.StringIO()
            with redirect_stderr(err):
                return torch.hub.load(
                    'sigsep/open-unmix-pytorch',
                    model_name,
                    target=target,
                    device=device,
                    pretrained=True
                )
            print(err.getvalue())
        except AttributeError:
            raise NameError('Model does not exist on torchhub')
            # assume model is a path to a local model_name direcotry
    else:
        # load model from disk
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = next(Path(model_path).glob("%s*.pth" % target))
        print('target_model_path', target_model_path)
        state = torch.load(
            target_model_path,
            map_location=device
        )

        max_bin = utils.bandwidth_to_max_bin(
            44100, # state['sample_rate']
            results['args']['nfft'],
            results['args']['bandwidth']
        )

        unmix = model.OpenUnmix(
            n_fft=results['args']['nfft'],
            n_hop=results['args']['nhop'],
            nb_channels=results['args']['nb_channels'],
            hidden_size=results['args']['hidden_size'],
            max_bin=max_bin
        )


        unmix.load_state_dict(state)
        unmix.stft.center = True
        unmix.eval()
        unmix.to(device)

        return unmix


def istft(X, rate=44100, n_fft=4096, n_hopsize=1024):
    t, audio = scipy.signal.istft(
        X / (n_fft / 2),
        rate,
        nperseg=n_fft,
        noverlap=n_fft - n_hopsize,
        boundary=True
    )
    return audio


def separate(
    audio,
    targets,
    model_name='umxhq',
    niter=1, softmask=False, alpha=1.0,
    residual_model=False, device='cpu',
    outdir=None, song_dir=None, song_name=None,
    duration=None, tar=None
):
    """
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float()
    audio_torch = torch.squeeze(audio_torch, 0)
    audio_torch = audio_to_stft(audio_torch)
    X =  audio_torch.detach().cpu().numpy()
    audio_torch = stft_to_spec(audio_torch)
    
    accom_midi = os.path.join(song_dir, f'{tar[1]}.txt')
    target_midi = os.path.join(song_dir, f'{tar[0]}.txt')

    mask_accom = midi_to_mask(audio_torch.permute(1, 0, 2)[0].numpy(), accom_midi, start_end=(0, duration))
    mask_target = midi_to_mask(audio_torch.permute(1, 0, 2)[0].numpy(), target_midi, start_end=(0, duration))
    mask_target = mask_target / (mask_target + mask_accom)
    
    x_filtered = mask_target * audio_torch.permute(1, 0, 2)[0].numpy()

    x_filtered = torch.tensor(np.expand_dims(x_filtered, 1))
    audio_torch = torch.unsqueeze(audio_torch, 0)
    x_filtered = torch.unsqueeze(x_filtered, 0)
    audio_torch, x_filtered = audio_torch.to(device), x_filtered.to(device)

    source_names = []
    V = []

    for j, target in enumerate(tqdm.tqdm(targets)):
        unmix_target = load_model(
            target=target,
            model_name=model_name,
            device=device
        )
        Vj = unmix_target(audio_torch, x_filtered).cpu().detach().numpy()
        if softmask:
            # only exponentiate the model if we use softmask
            Vj = Vj**alpha
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    # convert to complex numpy type
    X = X[..., 0] + X[..., 1]*1j
    X = X.transpose(2, 1, 0)

    if residual_model or len(targets) == 1:
        V = norbert.residual_model(V, X, alpha if softmask else 1)
        source_names += (['residual'] if len(targets) > 1
                         else ['accompaniment'])

    Y = norbert.wiener(V, X.astype(np.complex128), niter,
                       use_softmask=softmask)
    
    estimates = {}
    for j, name in enumerate(source_names):
        audio_hat = istft(
            Y[..., j].T,
            n_fft=unmix_target.stft.n_fft,
            n_hopsize=unmix_target.stft.n_hop
        )
        estimates[name] = audio_hat.T

    return estimates


def inference_args(parser, remaining_args):
    inf_parser = argparse.ArgumentParser(
        description=__doc__,
        parents=[parser],
        add_help=True,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    inf_parser.add_argument(
        '--softmask',
        dest='softmask',
        action='store_true',
        help=('if enabled, will initialize separation with softmask.'
              'otherwise, will use mixture phase with spectrogram')
    )

    inf_parser.add_argument(
        '--niter',
        type=int,
        default=1,
        help='number of iterations for refining results.'
    )

    inf_parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help='exponent in case of softmask separation'
    )

    inf_parser.add_argument(
        '--samplerate',
        type=int,
        default=44100,
        help='model samplerate'
    )

    inf_parser.add_argument(
        '--residual-model',
        action='store_true',
        help='create a model for the residual'
    )
    return inf_parser.parse_args()


def test_main(
    indir=None, samplerate=44100, niter=1, alpha=1.0,
    softmask=False, residual_model=False, model='umxhq',
    targets=('vocals', 'drums', 'bass', 'other'),
    outdir=None, start=0.0, duration=-1.0, no_cuda=False, tar=None, comb=None
):

    cuda_index = 2
    cuda_str = 'cuda:' + str(cuda_index)
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device(cuda_str if use_cuda else "cpu")

    indir = indir[0]

    song_dirs = sorted([song for song in os.listdir(indir) if comb in song])

    
    for song_dir in song_dirs:
        out_folder = os.path.join(outdir, song_dir)
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        
        song_name = song_dir
        song_dir = os.path.join(indir, song_dir)
        
        # handling an input audio path
        input_file = os.path.join(song_dir, 'mix.wav')

        if not os.path.isfile(input_file):
            print('NO MIX')
            continue

        info = sf.info(input_file)
        start = int(start * info.samplerate)
        # check if dur is none
        if duration > 0:
            # stop in soundfile is calc in samples, not seconds
            stop = start + int(duration * info.samplerate)
        else:
            # set to None for reading complete file
            stop = None

        audio, rate = sf.read(
            input_file,
            always_2d=True,
            start=start,
            stop=stop
        )

        if audio.shape[1] > 2:
            warnings.warn(
                'Channel count > 2! '
                'Only the first two channels will be processed!')
            audio = audio[:, :2]

        if rate != samplerate:
            # resample to model samplerate if needed
            audio = resampy.resample(audio, rate, samplerate, axis=0)

        dur = audio.shape[0] / rate

        estimates = separate(
            audio,
            targets=targets,
            model_name=model,
            niter=niter,
            alpha=alpha,
            softmask=softmask,
            residual_model=residual_model,
            device=device,
            outdir=outdir,
            song_dir=song_dir,
            song_name=song_name,
            duration=dur,
            tar=tar
        )

        if not outdir:
            model_path = Path(model)
            if not model_path.exists():
                output_path = Path(Path(input_file).stem + '_' + model)
            else:
                output_path = Path(
                    Path(input_file).stem + '_' + model_path.stem
                )
        else:
            output_path = Path(outdir) / Path(song_name)
        
        output_path.mkdir(exist_ok=True, parents=True)
        for target, estimate in estimates.items():
            sf.write(
                str(output_path / Path(target).with_suffix('.wav')),
                estimate,
                samplerate
            )


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(
        description='OSU Inference',
        add_help=False
    )

    parser.add_argument(
        'indir',
        type=str,
        nargs='+',
        help='paths to wav/flac folder.'
    )

    parser.add_argument(
        '--targets',
        nargs='+',
        default=['vocals'],
        type=str,
        help='provide targets to be processed. \
              If none, all available targets will be computed'
    )

    parser.add_argument(
        '--outdir',
        type=str,
        help='Results path where audio evaluation results are stored'
    )

    parser.add_argument(
        '--start',
        type=float,
        default=0.0,
        help='Audio chunk start in seconds'
    )

    parser.add_argument(
        '--duration',
        type=float,
        default=-1.0,
        help='Audio chunk duration in seconds, negative values load full track'
    )

    parser.add_argument(
        '--model',
        default='umxhq',
        type=str,
        help='path to mode base directory of pretrained models'
    )

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA inference'
    )

    args, _ = parser.parse_known_args()
    args = inference_args(parser, args)


    combs = ['ag_ag', 'ag_eg', 'eg_eg']
    tars = [['gt_1', 'gt_0'], ['gt_0', 'gt_1']]

    for comb in combs:
        for tar in tars:            
            test_main(
                indir=args.indir, samplerate=args.samplerate,
                alpha=args.alpha, softmask=args.softmask, niter=args.niter,
                residual_model=args.residual_model, model=args.model,
                targets=args.targets, outdir=args.outdir, start=args.start,
                duration=args.duration, no_cuda=args.no_cuda, tar=tar, comb=comb
            )