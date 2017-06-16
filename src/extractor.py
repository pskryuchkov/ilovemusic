#!/Users/pavel/anaconda/bin/python
#!/usr/bin/env python

import os
import time
import librosa
import re
import numpy as np
import wavio
import fnmatch
from classifier import load
from config import *
import os.path
import argparse, sys

def snorm(s):
    s = "".join([ c if c.isalnum() else "_" for c in s ])
    return re.sub('_+','_',s)

def file_extension(str):
    _, extension = os.path.splitext(str)
    return extension

def file_name(str):
    name, _ = os.path.splitext(str)
    return name

def rosaload(fn, dur):
    start = time.time()

    y, sr = librosa.load(fn, duration = dur)

    #print "loadtime:", time.time() - start
    #print y.shape
    return y, sr

def wvioload(fn, dur):
    #import scipy.io.wavfile as wavfile
    #fs, audiofile = wavfile.read(fn, False)
    start = time.time()

    fs, fc, audiofile = wavio.readwav(fn)

    y = audiofile.T[0]
    y = y[:(fs * dur)]  #FIXME
    y = y * 1.0

    #print "loadtime:", time.time() - start
    #print y.shape
    return y, fs

def ext_bpm(y, sr):
    #y, sr = loadfunc(fn, dur)
    start = time.time()

    #tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    #onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate = np.median)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    aggregate = np.median,
    fmax = 8000, n_mels = 256)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr = sr)


    #print "functime:", time.time() - start
    return tempo
"""
def ext_bpm_new(path, loadfunc, dur = config.duration):
    #get_file_bpm(path, params = None):
    params = None
    from aubio import source, tempo
    from numpy import median, diff
    #Calculate the beats per minute (bpm) of a given file.
    #    path: path to the file
    #    param: dictionary of parameters

    if params is None:
        params = {}
    try:
        win_s = params['win_s']
        samplerate = params['samplerate']
        hop_s = params['hop_s']
    except KeyError:

        # super fast
        samplerate, win_s, hop_s = 4000, 128, 64
        # fast
        samplerate, win_s, hop_s = 8000, 512, 128

        # default:
        samplerate, win_s, hop_s = 44100, 1024, 512

    s = source(path, samplerate, hop_s)
    samplerate = s.samplerate
    o = tempo("specdiff", win_s, hop_s, samplerate)
    # List of beats, in samples
    beats = []
    # Total number of frames read
    total_frames = 0

    while True:
        samples, read = s()
        is_beat = o(samples)
        if is_beat:
            this_beat = o.get_last_s()
            beats.append(this_beat)
            #if o.get_confidence() > .2 and len(beats) > 2.:
            #    break
        total_frames += read
        if read < hop_s:
            break

    # Convert to periods and to bpm
    if len(beats) > 1:
        if len(beats) < 4:
            print("few beats found in {:s}".format(path))
        bpms = 60./diff(beats)
        b = median(bpms)
    else:
        b = 0
        print("not enough beats found in {:s}".format(path))
    return b
"""
def dissmeasure(fvec, amp):
    fvec = np.asarray(fvec)
    amp = np.asarray(amp)

    Dstar = 0.24
    S1, S2 = 0.0207, 18.96
    C1, C2 = 5, -5
    A1, A2 = -3.51, -5.75

    ams = amp[np.argsort(fvec)]
    fvec = np.sort(fvec)
    D = 0
    for i in range(1, len(fvec)):
        Fmin = fvec[:-i]
        S = Dstar / (S1 * Fmin + S2)
        Fdif = fvec[i:] - fvec[:-i]

        a = np.minimum(ams[:-i], ams[i:]) # min model

        Dnew = a * (C1 * np.exp(A1 * S * Fdif) + C2 * np.exp(A2 * S * Fdif))
        D += np.sum(Dnew)

    return D

def ext_coss(y, sr):

    start = time.time()
    spec = np.flipud(np.abs(librosa.stft(y))).T

    slen, swide = spec.shape
    maximums = []
    cs = []
    for j in range(slen):
        top = spec[j].argsort()[-10:][::-1]
        fv = []
        av = []
        for k in range(len(top)):
            fv.append(top[k] * 10.756)
            av.append(spec[j,k])
            maximums.append([j, -top[k], spec[j,k]])

        cs.append(dissmeasure(fv, av))

    csf = np.asarray(cs)
    #print "functime:", time.time() - start

    return np.mean(csf[:-2])

def ext_harmony(y, sr):

    start = time.time()
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    #print "functime:", time.time() - start

    return np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y_harmonic))
                                         + np.sum(np.abs(y_percussive)))

def ext_centroid(y, sr):
    #y, sr = loadfunc(fn, dur)

    start = time.time()
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    #print "functime:", time.time() - start

    return cent

def ext_onset(y, sr):

    start = time.time()
    o_times = librosa.frames_to_time(
                librosa.onset.onset_detect(
                    onset_envelope=librosa.onset.onset_strength(y, sr=sr), sr=sr), sr=sr)

    delta = o_times[1:] - o_times[:-1]
    h = np.histogram(delta, bins = np.arange(0, 3, step=0.02))[0]
    h = 1.0 * h[:30] / sum(h)
    #print "functime:", time.time() - start

    return h


def round3(x):
    return "%.3f" % round(x, 3)


def spectrum_filter(S, low_idx = 15, high_idx = 15):
    S[:low_idx, :] = 0
    S[-high_idx:, :] = 0
    return S

def chroma_smooth(chroma, n = 10, filter = np.mean, filter_edges = True):
    chroma_filtered = np.zeros([len(chroma),len(chroma[0])])
    for i in range(len(chroma)):
        for j in range(n, len(chroma[i])):
            chroma_filtered[i,j] = filter(chroma[i, j-n:j+n])
    return chroma_filtered

def chroma_smooth_new(chroma, n = 12):
    #print chroma.shape
    chroma_filtered = []
    for k in range(len(chroma)):
        chroma_filtered.append([1.0 * sum(chroma[k][i:i+n])/n
                                for i in range(0,len(chroma[k]),n)])
    #print len(chroma[0]), len(chroma_filtered[0])
    return chroma_filtered

def ext_chords(y, sr, check_plato = False):
    start = time.time()
    S = np.abs(librosa.stft(y, n_fft=1024))
    #S = spectrum_filter(S)

    #chroma = chroma_smooth_new(librosa.feature.chroma_stft(S=S, sr=sr).T[0::2].T, filter = np.median)
    chroma = chroma_smooth_new(librosa.feature.chroma_stft(S=S, sr=sr))
    chroma = np.array(chroma)
    #            c c#  d  d# e  f  f# g  g# a  a# b
    chords =   [[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], # C       1
                [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], # C#      2
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0], # D       3
                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], # D#      4
                [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], # E       5
                [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], # F       6
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0], # F#      7
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1], # G       8
                [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], # G#      9
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], # A       10
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0], # A#      11
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1], # B       12
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], # c       13
                [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], # c#      14
                [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0], # d       15
                [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0], # d#      16
                [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], # e       17
                [1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], # f       18
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0], # f#      19
                [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0], # g       20
                [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], # g#      21
                [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], # a       22
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], # a#      23
                [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]] # b       24

    chord_dict = {  0: "C", 1:"C#", 2: "D", 3:"D#",  4:"E",  5:"F",  6:"F#",  7:"G", 8:"G#", 9: "A", 10:"A#", 11: "B",
                    12: "c", 13:"c#", 14:"d", 15:"d#", 16:"e", 17:"f", 18:"f#", 19:"g", 20:"g#", 21: "a", 22:"a#", 23:"b"}

    melody_chords = []
    for cvec in chroma.T:
        sample_matrix = np.tile(np.array(cvec), (24, 1))
        melody_chords.append(np.argmax(np.sum(sample_matrix * chords, axis=1)))

    if check_plato:
        n_plato = 0
        for i in range(1,len(melody_chords)):
            if melody_chords[i] == melody_chords[i-1]: n_plato += 1
        print "n_plato", n_plato

    h = np.histogram(melody_chords, bins = 24)[0]

    #print "functime:", time.time() - start
    return h.tolist()

def ext_volume(y, sr):
    """
    y, sr = loadfunc(fn, dur)

    S = np.abs(librosa.stft(y))
    vol = np.mean(librosa.core.logamplitude(S ** 2))
    print vol
    print "functime:", time.time() - start
    return vol
    """
    start = time.time()
    S = np.abs(librosa.stft(y))
    #print "functime:", time.time() - start
    return np.mean(librosa.core.logamplitude(S ** 2))


def ext_psnr(y, sr):
    start = time.time()
    S = np.abs(librosa.stft(y))
    #psnr = [np.mean(np.abs(frame)) / np.std(np.abs(frame)) for frame in S[0]]
    #vol = np.mean(np.abs(y)) / np.std(np.abs(y))
    ls = S ** 2
    psnr = [np.mean(np.abs(frame)) / np.std(np.abs(frame)) for frame in ls.T]
    #vol = np.mean(ls) / np.std(ls)
    vol = np.mean(psnr)
    #print "functime:", time.time() - start
    return vol

def ext_correlation(y, sr):
    yr = librosa.resample(y, sr, 11025)
    start = time.time()
    corr = librosa.autocorrelate(yr)
    #print "functime:", time.time() - start
    return np.std(corr)

def ext_onset_strength(y, sr):
    o_env = librosa.onset.onset_strength(y, sr=sr)
    o_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    return np.mean(o_env[o_frames])

def ext_zero_cross(y, sr):
    z = librosa.zero_crossings(y)
    return 1.0 * z[z == True].shape[0] / z.shape[0]

def ext_flux(y, sr):
    S = np.abs(librosa.stft(y)).T
    quad_flux = np.mean((S[1:] - S[:-1]) ** 2, axis = 1)
    return np.mean(quad_flux)

def ext_contrast(y, sr):
    S = np.abs(librosa.stft(y)).T
    n_edge = 200
    contrast = []
    for spec in S:
        sorted_spec = np.sort(spec)
        bigger = np.mean(sorted_spec[-n_edge:])
        smaller = np.mean(sorted_spec[:n_edge])
        contrast.append(bigger - smaller)
    return np.mean(contrast)

def ext_onset_regular(y, sr):
    o_times = librosa.frames_to_time(
                librosa.onset.onset_detect(
                    onset_envelope=librosa.onset.onset_strength(y, sr=sr), sr=sr), sr=sr)

    delta = o_times[1:] - o_times[:-1]
    return np.std(delta) / np.mean(delta)

def ext_spec_std(y, sr):
    S = np.abs(librosa.stft(y)).T
    spec_std = np.std(S, axis = 1)
    return np.mean(spec_std)

def ext_spec_median(y, sr):
    S = np.abs(librosa.stft(y)).T
    spec_median = np.median(S, axis=1)
    return np.mean(spec_median)

def ext_spec_rolloff(y, sr):
    return np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95))

def ext_spec_bandwidth(y, sr):
    return np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

def file_list():
    # FIXME
    # way 1
    #return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    # way 2
    filelist = []
    for root, dirnames, filenames in os.walk(project_path):
        for filename in fnmatch.filter(filenames, '*.mp3'):
            filepath = os.path.join(root, filename)
            if filepath.find("/top/") > -1:
                filelist.append(filepath)
    return filelist


def get_info(str):
    return [snorm(file_name(os.path.basename(project_path + str))),
            snorm(str.split("/")[-3].split("-")[0].strip())]


def open_csv_write(fn):
    file = open(fn + ".csv", 'w+')
    file.write("sep=,\n")
    return file


def do_process(feature_list, func_base, sort_type='value', check_base=True):
    #features_path = "../data/features/basic/"
    #features_path = "../data/features/experimental/"
    features_path = basepath
    filelist = file_list()

    n_features = len(feature_list)
    data = [[] for p in range(n_features)]

    base_songs = []
    if check_base:
        base_songs = [x[1:] for x in load(features_path +
                                          feature_list[0] + ".csv")]

    for k, song_file in enumerate(filelist[:50]):
        ext = file_extension(song_file)
        if ext in music_ext:
            song, artist = get_info(song_file)
            if (check_base and not [song, artist] in base_songs) \
                    or not check_base:

                y, sr = rosaload(song_file, 60)
                print "{}. {}, {}".format(k+1, song, artist)
                for p, feature in enumerate(feature_list):
                    y_dur = y[:sr * durs[feature]]
                    feature_value = func_base[feature](y_dur, sr)
                    #print feature, feature_value
                    data[p].append([feature_value, song, artist])
            else:
                print "Already Present:", song, artist

    for p in range(len(feature_list)):
        base = []
        if check_base and os.path.isfile(features_path + feature_list[p] + ".csv") :
            base = load(features_path + feature_list[p] + ".csv")
        data[p] += base

        stat = open_csv_write(features_path + feature_list[p])
        sub_data = None
        if sort_type == 'value':
            sub_data = sorted(data[p], key=lambda x: (x[0]))
        elif sort_type == 'artist':
            sub_data = sorted(data[p], key=lambda x: (x[2]))

        for record in sub_data:
            val = None
            if type(record[0]) is float: val = round3(record[0])
            elif type(record[0]) is list:
                val = "; ".join(map(lambda x: str(x), record[0]))

            stat.write("{0},{1},{2}\n".format(record[2],
                                              record[1],
                                              val))
        stat.close()


def refresh_base(check_base):
    do_process(basic_funcs.keys(), basic_funcs, check_base=check_base)


def rebuild_base():
    refresh_base(False)


def update_base():
    refresh_base(True)


def arg_run():
    # extractor.py
    # -e <feature>          calculate feature, rewrite feature file
    # -u                    update base (add new songs)
    # -r                    rebuild base (recalculate all features)
    # config.base_path      path with basic features
    # config.complex_path   path with composition features

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-e', nargs=1)
    parser.add_argument('-u', action='store_true')
    parser.add_argument('-r', action='store_true')
    args = parser.parse_args()

    if sum(map(bool, [args.e, args.u, args.r])) > 1:
        print "Error: too many arguments"
        exit()

    if args.e is not None:
        feature = args.e[0]
        if feature in basic_funcs.keys():
            do_process([args.e[0]], basic_funcs, check_base=False)
        else:
            do_process([args.e[0]], exp_funcs, check_base=False)

    elif args.u is True:
        update_base()

    elif args.r is True:
        rebuild_base()

basic_funcs = {
            'bpm': ext_bpm,    # FIXME
            'centroid': ext_centroid,
            'volume': ext_volume,
            'self-correlation': ext_correlation,
            'zero_cross': ext_zero_cross,
            'onset_strength': ext_onset_strength,
            'onset_regular': ext_onset_regular,
            'spectral_flux': ext_flux,
            'spectral_contrast': ext_contrast,
            'spectral_median': ext_spec_median,
            'spectral_std': ext_spec_std,
            'spectral_bandwidth': ext_spec_bandwidth,
            'spectral_rolloff': ext_spec_rolloff
        }

exp_funcs = {
            'chords': ext_chords,
            'coss': ext_coss,
            'harmony': ext_harmony,
            'psnr': ext_psnr
        }

feature_names = basic_funcs.keys()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg_run()
    else:
        pass
        #do_process(basic_funcs.keys(), check_base=False)
        #do_process(["chords"], exp_funcs, check_base=False)
        #rebuild_base()
        #update_base()


