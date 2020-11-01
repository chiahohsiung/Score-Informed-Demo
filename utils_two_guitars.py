import shutil
import torch
import os
import numpy as np
import warnings
from bisect import bisect_right, bisect_left


def _sndfile_available():
    try:
        import soundfile
    except ImportError:
        return False

    return True


def _torchaudio_available():
    try:
        import torchaudio
    except ImportError:
        return False

    return True


def get_loading_backend():
    if _torchaudio_available():
        return torchaudio_loader

    if _sndfile_available():
        return soundfile_loader


def get_info_backend():
    if _torchaudio_available():
        return torchaudio_info

    if _sndfile_available():
        return soundfile_info


def soundfile_info(path):
    import soundfile
    info = {}
    sfi = soundfile.info(path)
    info['samplerate'] = sfi.samplerate
    info['samples'] = int(sfi.duration * sfi.samplerate)
    info['duration'] = sfi.duration
    return info


def soundfile_loader(path, start=0, dur=None):
    import soundfile
    # get metadata
    info = soundfile_info(path)
    start = int(start * info['samplerate'])
    # check if dur is none
    if dur:
        # stop in soundfile is calc in samples, not seconds
        stop = start + int(dur * info['samplerate'])
    else:
        # set to None for reading complete file
        stop = dur

    audio, _ = soundfile.read(
        path,
        always_2d=False,
        start=start,
        stop=stop
    )
    return torch.FloatTensor(audio.T)


def torchaudio_info(path):
    import torchaudio
    # get length of file in samples
    info = {}
    si, _ = torchaudio.info(str(path))
    info['samplerate'] = si.rate
    info['samples'] = si.length // si.channels
    info['duration'] = info['samples'] / si.rate
    return info


def torchaudio_loader(path, start=0, dur=None):
    import torchaudio
    info = torchaudio_info(path)
    # loads the full track duration
    if dur is None:
        sig, rate = torchaudio.load(path)
        return sig
        # otherwise loads a random excerpt
    else:
        num_frames = int(dur * info['samplerate'])
        offset = int(start * info['samplerate'])
        sig, rate = torchaudio.load(
            path, num_frames=num_frames, offset=offset
        )
        return sig


def load_info(path):
    loader = get_info_backend()
    return loader(path)


def load_audio(path, start=0, dur=None):
    loader = get_loading_backend()
    return loader(path, start=start, dur=dur)


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def save_checkpoint(
    state, is_best, path, target
):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

            
def slicefft_slices(pitch, size, interval=30, tuning_freq=440,nharmonics=20,fmin=25,fmax=18000,iscale = 'lin',sampleRate=44100):
    if pitch > 0:
        binfactor = float(size) / float(sampleRate)
        # print('binfactor', binfactor )
        fups,fdowns = getfreqs(pitch,interval=interval,tuning_freq=tuning_freq,nharmonics=nharmonics)
        # print('fups', fups)
        # print('\n')
        # print('fdowns', fdowns)
        # exit()

        ranges = tuple((1+int(np.floor(fdowns[f]*binfactor)), 1+int(np.ceil(fups[f]*binfactor))) for f in range(len(fdowns)))
        # print('ranges', ranges)
        ranges = remove_overlap(ranges)
        # print('ranges', ranges)
        slices_y = [slice(ranges[f][0],ranges[f][1]) for f in range(len(ranges)) if ranges[f][1]<=(size/2+1)]
        # print('slices_y', slices_y)
        # exit()
    else:
        slices_y = []
    return slices_y


# read a txt containing midi notes : onset, offset, midinote
def expandMidi(FilePath, beginTime, finishTime, interval, tuning_freq,nharmonics,samplerate,hop,window,timeSpan_on,timeSpan_off,nframes,fermata=0.):
    fermata = np.maximum(timeSpan_off,fermata)
    
    # Read the midi txt
    onsets, offsets, pitch = np.genfromtxt(FilePath, unpack=True, skip_header=1, usecols=(0, 1, 3))
    melTimeStampsBeginO = onsets.tolist()
    melTimeStampsEndO = offsets.tolist()
    
    # Get the index of the begining and finishing of the desired period 
    # Change startTime to startIndex, same as endTime
    startTime = bisect_right(melTimeStampsEndO,beginTime)
    endTime = bisect_left(melTimeStampsBeginO,finishTime)

    if melTimeStampsEndO[startTime]<float(beginTime):
        startTime = startTime + 1
    if endTime >= len(melTimeStampsBeginO):
        endTime = len(melTimeStampsBeginO) - 1
    elif melTimeStampsBeginO[endTime] > float(finishTime):
        endTime = endTime - 1

    if (startTime < endTime):
        melTimeStampsBeginO = melTimeStampsBeginO[startTime:endTime + 1]
        melTimeStampsBegin = [x - beginTime for x in melTimeStampsBeginO]
        melTimeStampsEndO = melTimeStampsEndO[startTime:endTime + 1]
        melTimeStampsEnd = [x - beginTime for x in melTimeStampsEndO]

        for i in range(len(melTimeStampsBegin)):
            if (melTimeStampsBegin[i] < 0):
                melTimeStampsBegin[i] = 0.0
            if (melTimeStampsEnd[i] < 0):
                melTimeStampsEnd[i] = 0.0
            if (melTimeStampsEnd[i] > (finishTime-beginTime)):
                melTimeStampsEnd[i] = finishTime - beginTime
            if (melTimeStampsBegin[i] > (finishTime-beginTime)):
                melTimeStampsBegin[i] = finishTime - beginTime

        #get the midi
        melNotesMIDI = pitch.tolist()
        melNotesMIDI = melNotesMIDI[startTime:endTime+1]
        melIndex=[k for k in range(startTime,endTime+1)]
         # Why no minus window length
        tframes = float(nframes) * float(hop) / float(samplerate)
        #eliminate short notes
        lenlist = len(melTimeStampsBegin)
        i = 0
        while i < lenlist:
            if (melTimeStampsEnd[i]<=0) \
                or (melTimeStampsEnd[i]<=melTimeStampsBegin[i]) \
                or (melTimeStampsBegin[i]>=tframes) \
                or ((melTimeStampsEnd[i] - melTimeStampsBegin[i])<0.01) :
                melTimeStampsBegin.pop(i)
                melTimeStampsEnd.pop(i)
                melNotesMIDI.pop(i)
                melIndex.pop(i)
                lenlist=lenlist-1
                i=i-1
            i=i+1

        maxAllowed_on = int(round(timeSpan_on * float(samplerate / hop)))
        maxAllowed_off = int(round(timeSpan_off * float(samplerate / hop)))

        endMelody = int((finishTime - beginTime) * round(float(samplerate / hop)))

        w = window / 2 / hop

        melodyBegin = []
        melodyEnd = []
        for i in range(len(melTimeStampsEnd)):
            melodyBegin.append(np.maximum(0,int(melTimeStampsBegin[i] * round(float(samplerate / hop))) - maxAllowed_on))
            intersect = [mb for mb,me in zip(melTimeStampsBegin,melTimeStampsEnd) if (mb>melTimeStampsBegin[i]) and (me+timeSpan_off)>=(melTimeStampsBegin[i]-timeSpan_on) and (mb-timeSpan_on)<=(melTimeStampsEnd[i]+timeSpan_off) ]
            if len(intersect)==0:
                notesafter = filter(lambda x: (x-timeSpan_on)>(melTimeStampsEnd[i] +timeSpan_off), melTimeStampsBegin)
                
                notesafter_ls = list(notesafter)
                if len(notesafter_ls)>0:
                    #import pdb;pdb.set_trace()
                    newoffset= np.minimum(melTimeStampsEnd[i] + fermata, np.maximum(0,min(notesafter_ls)-timeSpan_on) )
                else:
                    newoffset= melTimeStampsEnd[i] + fermata
                melodyEnd.append(np.minimum(nframes,np.minimum(endMelody,int(newoffset * round(float(samplerate / hop))))))
            else:
                melodyEnd.append(np.minimum(nframes,np.minimum(endMelody,int(melTimeStampsEnd[i] * round(float(samplerate / hop))) + maxAllowed_off)))

        melNotes = melNotesMIDI
        intervals = np.zeros((len(melNotes), 2 * nharmonics + 3))
        for i in range(len(melNotes)):
            intervals[i, 0] = melodyBegin[i]
            intervals[i, 1] = melodyEnd[i]
            intervals[i, 2] = melNotes[i]
            slice_y = slicefft_slices(melNotes[i], size=window, interval=interval, tuning_freq=tuning_freq,nharmonics=nharmonics,sampleRate=samplerate)
            tmp = [slice_y[j].start for j in range(len(slice_y))]
            
            intervals[i, 3:2*len(tmp)+3:2] = tmp            
            intervals[i, 4:2*len(tmp)+4:2] = [slice_y[k].stop for k in range(len(slice_y))]
        return intervals


# Return the lenght in seconds of the midi
def getMidiLength(instrument, FilePath):
    midifile = os.path.join(FilePath, instrument + '.txt')
    melodyFromFile = np.genfromtxt(midifile, comments='!', \
      delimiter=',',names="a,b,c",dtype=["f","f","S3"])
    melTimeStampsBeginO = melodyFromFile['a'].tolist()
    melTimeStampsEndO = melodyFromFile['b'].tolist()
    return max(melTimeStampsEndO)


def getfreqs(midinote,interval=30,tuning_freq=440,nharmonics=20,ismidi=True):
    factor = 2.0**(interval/1200.0)
    if ismidi:
        f0 = float(midi2freq(midinote,tuning_freq=tuning_freq))
    else:
        f0 = midinote
    fdowns = [f * f0 / float(factor) for f in range(1,nharmonics)]
    fups = [f * f0 * float(factor) for f in range(1,nharmonics)]
    return fups,fdowns


# Useful constants
MIDI_A4 = 69   # MIDI Pitch number
FREQ_A4 = 440. # Hz
SEMITONE_RATIO = 2. ** (1. / 12.) # Ascending
def midi2freq(midi_number, tuning_freq=440., MIDI_A4=69.):
    return float(tuning_freq) * 2.0 ** ((float(midi_number) - float(MIDI_A4)) * (1./12.))


#removes overlap between two intervals
def remove_overlap(ranges):
    result = []
    current_start = -1
    current_stop = -1

    for start, stop in sorted(ranges):
        if start > current_stop:
            # this segment starts after the last segment stops
            # just add a new segment
            result.append( (start, stop) )
            current_start, current_stop = start, stop
        else:
            # segments overlap, replace
            result[-1] = (current_start, stop)
            # current_start already guaranteed to be lower
            current_stop = max(current_stop, stop)

    return result

def getMidiNum(FilePath, beginTime, finishTime):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            
            onsets, offsets, pitch = np.genfromtxt(FilePath, unpack=True, skip_header=1, usecols=(0, 1, 3))
        
        melTimeStampsBeginO = onsets.tolist()
        melTimeStampsEndO = offsets.tolist()
        startTime = bisect_right(melTimeStampsEndO, beginTime)
        endTime = bisect_left(melTimeStampsBeginO, finishTime)



        if melTimeStampsEndO[startTime]<float(beginTime):
            startTime=startTime+1
        if endTime>=len(melTimeStampsBeginO):
            endTime = len(melTimeStampsBeginO) - 1
        elif melTimeStampsBeginO[endTime]>float(finishTime):
            endTime=endTime-1

        if (startTime<endTime):
            melTimeStampsBeginO = melTimeStampsBeginO[startTime:endTime+1]
            melTimeStampsBegin = [x - beginTime for x in melTimeStampsBeginO]
            melTimeStampsEndO = melTimeStampsEndO[startTime:endTime+1]
            melTimeStampsEnd = [x - beginTime for x in melTimeStampsEndO]

            for i in range(len(melTimeStampsBegin)):
                if (melTimeStampsBegin[i] < 0):
                    melTimeStampsBegin[i] = 0.0
                if (melTimeStampsEnd[i] < 0):
                    melTimeStampsEnd[i] = 0.0
                if (melTimeStampsEnd[i] > (finishTime-beginTime)):
                    melTimeStampsEnd[i] = finishTime-beginTime
                if (melTimeStampsBegin[i] > (finishTime-beginTime)):
                    melTimeStampsBegin[i] = finishTime-beginTime

            #get the midi
            melNotesMIDI = pitch.tolist()
            melNotesMIDI = melNotesMIDI[startTime:endTime+1]
            melIndex=[k for k in range(startTime,endTime+1)]
            #eliminate short notes
            lenlist = len(melTimeStampsBegin)
            i=0
            while i<lenlist:
                if (melTimeStampsEnd[i]<=0) \
                    or (melTimeStampsEnd[i]<=melTimeStampsBegin[i]) \
                    or ((melTimeStampsEnd[i]-melTimeStampsBegin[i])<0.01) :
                    melTimeStampsBegin.pop(i)
                    melTimeStampsEnd.pop(i)
                    melNotesMIDI.pop(i)
                    melIndex.pop(i)
                    lenlist=lenlist-1
                    i=i-1
                i=i+1
            return len(melNotesMIDI)
        else:
            return 1
    except:
        return 1

def filterSpec(mag,notes,ninst,start,stop,timbre_model_path=None):
    if timbre_model_path is not None:
        with open(timbre_model_path, 'rb') as input:
            harmonics= pickle.load(input)
    filtered = np.ones((ninst,mag.shape[0],mag.shape[1]), dtype=np.float32) * 1e-18
    for j in range(ninst): #for all the inputed instrument notes
        for p in range(len(notes[j])): #for all notes
            if notes[j,p,2] > 0 and np.maximum(0, np.minimum(notes[j,p,1], stop) - np.maximum(notes[j,p,0], start))>0:
                begin = int(np.maximum(notes[j,p,0], start))-start
                end = int(np.minimum(notes[j,p,1], stop))-start
                slice_x = slice(begin,end,None)
                slices_y_start = notes[j,p,3::2]
                slices_y_stop = notes[j,p,4::2]
                if timbre_model_path is None:
                    slices_y = np.hstack(tuple([range(int(slices_y_start[f]),int(slices_y_stop[f])) for f in range(np.minimum(len(slices_y_start),len(slices_y_stop))) if slices_y_stop[f]>0]))
                    filtered[j,slice_x,slices_y] = 1.
                    slices_y = None
                else:
                    for k in range(len(slices_y_start)):
                        filtered[j,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] = filtered[j,slice_x,slice(int(slices_y_start[k]),int(slices_y_stop[k]),None)] + harmonics[j,int(notes[j,p,2]),k]
                slice_x = None
    mask = np.zeros((mag.shape[0], ninst*mag.shape[1]), dtype=np.float32)
    for j in range(ninst): #for all the inputed instrument pitches
        mask[:,j*mag.shape[1]:(j+1)*mag.shape[1]] = filtered[j,:,:] / np.max(filtered[j,:,:])
    filtered = None
    j = None
    p = None
    f = None
    return mask


def midi_to_mask(mix, target_midi, start_end=(0, 10.0)):
    nharmonics = 20
    interval = 30 #cents
    tuning_freq = 440 #Hz
    sr = 44100
    hop = 1024
    frameSize = 4096
    start = start_end[0]
    end = start_end[1]
    T, F = mix.shape
    ending = int((T * hop + frameSize) / sr)
    nelem_g = 1
    ng = getMidiNum(target_midi, start, end)
    nelem_g = np.maximum(ng, nelem_g)

    if int(nelem_g) == 1:
        masks_temp = np.ones((mix.shape[0], mix.shape[1]), dtype=np.float32) * 1e-18
    else:
        melody = np.zeros((1, int(nelem_g), 2*nharmonics+3))
        nframes = int(44100 * end / np.double(hop)) + 2
        
        tmp = expandMidi(target_midi, start, end, interval, tuning_freq, nharmonics, sr, hop, frameSize, 0.1, 0.1, nframes,0.5)
        melody[0,:tmp.shape[0],:] = tmp
        tmp = None
        jump = mix.shape[0]
        masks_temp = filterSpec(mix, melody, 1, 0, nframes)

    return masks_temp

