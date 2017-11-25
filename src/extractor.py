#!/usr/bin/env python

from os.path import isfile, realpath, dirname, exists
from os import chdir, makedirs
from config import *
import argparse, sys
import numpy as np
import librosa
import fnmatch
import os.path
import os
import re


def norm_fn(s):
    s = "".join([ c if c.isalnum() else "_" for c in s ])
    return re.sub('_+','_',s)


def file_extension(str):
    _, extension = os.path.splitext(str)
    return extension


def file_name(str):
    name, _ = os.path.splitext(str)
    return name


def rosaload(fn, offset, dur):
    y, sr = librosa.load(fn, offset=offset, duration = dur)
    return y, sr


def spectrum_filter(S, low_idx=15, high_idx=15):
    S[:low_idx, :] = 0
    S[-high_idx:, :] = 0
    return S


def chroma_smooth(chroma, n=12):
    chroma_filtered = []
    for k in range(len(chroma)):
        chroma_filtered.append([1.0 * sum(chroma[k][i:i+n])/n
                                for i in range(0,len(chroma[k]),n)])
    return chroma_filtered


def e_chords(y, sr, check_plato = False):
    S = np.abs(librosa.stft(y, n_fft=1024))
    # S = spectrum_filter(S)

    chroma = chroma_smooth(librosa.feature.chroma_stft(S=S, sr=sr))
    chroma = np.array(chroma)
    #            c c#  d  d# e  f  f# g  g# a  a# b
    chords =    [[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], # C       1
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
        print("n_plato{}".format(n_plato))

    h = np.histogram(melody_chords, bins = 24)[0]

    return h.tolist()


def e_bpm(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr,
        aggregate=np.median, fmax=8000, n_mels=256)

    dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr,
                                aggregate = None)

    return np.median(dtempo), np.std(dtempo)


def e_harmony(y, sr):
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    return np.sum(np.abs(y_harmonic)) / (np.sum(np.abs(y_harmonic))
                                         + np.sum(np.abs(y_percussive)))


def e_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid), np.std(centroid)


def e_volume(y, sr):
    S = np.abs(librosa.stft(y))
    amplitude = librosa.core.logamplitude(S ** 2)
    return np.mean(amplitude), np.std(amplitude)


def e_correlation(y, sr):
    yr = librosa.resample(y, sr, 11025)
    corr = librosa.autocorrelate(yr)
    return np.std(corr)


def e_onset_strength(y, sr):
    o_env = librosa.onset.onset_strength(y, sr=sr)
    o_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
    s = o_env[o_frames]
    return np.mean(s), np.std(s)


def e_zero_cross(y, sr):
    z = librosa.zero_crossings(y)
    return 1.0 * z[z == True].shape[0] / z.shape[0]


def e_flux(y, sr):
    S = np.abs(librosa.stft(y)).T
    quad_flux = np.mean((S[1:] - S[:-1]) ** 2, axis = 1)
    return np.mean(quad_flux), np.std(quad_flux)


def e_contrast(y, sr):
    S = np.abs(librosa.stft(y)).T
    n_edge = 200
    contrast = []
    for spec in S:
        sorted_spec = np.sort(spec)
        bigger = np.mean(sorted_spec[-n_edge:])
        smaller = np.mean(sorted_spec[:n_edge])
        contrast.append(bigger - smaller)
    return np.mean(contrast), np.std(contrast)


def e_onset_regular(y, sr):
    o_times = librosa.frames_to_time(
                librosa.onset.onset_detect(
                    onset_envelope=librosa.onset.onset_strength(y, sr=sr), sr=sr), sr=sr)

    delta = o_times[1:] - o_times[:-1]
    return np.std(delta) / np.mean(delta)


def e_spec_std(y, sr):
    S = np.abs(librosa.stft(y)).T
    spec_std = np.std(S, axis = 1)
    return np.mean(spec_std), np.std(spec_std)


def e_spec_median(y, sr):
    S = np.abs(librosa.stft(y)).T
    spec_median = np.median(S, axis=1)
    return np.mean(spec_median), np.std(spec_median)


def e_spec_rolloff(y, sr):
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)
    return np.mean(rolloff), np.std(rolloff)


def e_cepstral(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return np.mean(mfccs, axis=1), np.std(mfccs, axis=1)


def e_spec_bandwidth(y, sr):
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.mean(bandwidth), np.std(bandwidth)


def song_list(songs_directory):
    result = []
    for root, dirnames, filenames in os.walk(songs_directory):
        # FIXME: file extensions
        for filename in fnmatch.filter(filenames, '*.mp3'):
            filepath = os.path.join(root, filename)
            result.append(filepath)
    return result


# extract song name and artist name from path
# <artist_name>/<song_name>
# <artist_name> - <album_name>/<song_name>
def song_info(songs_directory, str):
    return [norm_fn(file_name(os.path.basename(songs_directory + str))),
            norm_fn(str.split("/")[-2].split("-")[0].strip())]


def open_csv_write(fn):
    file = open(fn + ".csv", 'w+')
    file.write("sep=,\n")
    return file


def load_stat(fn):
    content = open(fn, 'r').readlines()
    # skip csv header
    content = content[1:]

    data = []
    for line in content:
        vec = line.split(",")
        data.append([vec[0].rstrip(), vec[1].rstrip(), tuple(map(lambda x: float(x), vec[2:]))])
    return data


def load_csv(fn, sep=","):
    content = open(fn, 'r').readlines()
    content = content[1:]  # remove first line
    return [line.split(sep) for line in content]


# FIXME
def trim_track(y, sr, dur):
    return y[:sr * dur]


def do_process(mode, feature_list, func_base, check_base=True, n_tracks=None):
    min_song_len = 30
    offset = 60
    duration = 60

    features_path = None
    if mode == "favourite":
        songs_directory = favourite_path
        features_path = fav_songs_stat
    elif mode == "tag":
        songs_directory = tag_path
        features_path = tag_songs_stat

    files = song_list(songs_directory)

    base_songs = []
    if check_base:
        base_songs = [x[0:2] for x in load_csv(features_path +
                                              feature_list[0] + ".csv")]

    print("target: {} songs".format(mode))
    print("total: {0} tracks".format(len(files)))

    n_features = len(feature_list)
    data = [[] for p in range(n_features)]

    # feature extracting
    for k, song_file in enumerate(files[:n_tracks]):
        song, artist = song_info(songs_directory, song_file)

        if (check_base and not [artist, song] in base_songs) \
                or not check_base:

            y, sr = rosaload(song_file, offset=offset, dur=duration)

            if y.shape[0] < sr * min_song_len:
                print("track {} too short".format(song_file))
                continue

            print("{}. {}, {}".format(k + 1, song, artist))

            for p, feature in enumerate(feature_list):
                y_dur = trim_track(y, sr, durs[feature])
                e_feature_func = func_base[feature]
                feature_value = e_feature_func(y_dur, sr)

                data[p].append([artist, song, feature_value])
        else:
            print("already present: {}, {}".format(song, artist))

    # save results
    for p in range(len(feature_list)):
        if check_base and os.path.isfile(features_path +
                                            feature_list[p] + ".csv"):

            data[p] = load_stat(features_path +
                                feature_list[p] + ".csv") +\
                                data[p]

        write_stat(features_path, feature_list[p], data[p])


# FIXME
def is_float(var):
    e_type = type(var)
    return ((e_type is float) or \
           (e_type is np.float32) or \
           (e_type is np.float64))


# FIXME
def is_tuple(var):
    return (type(var) is tuple)


def write_stat(base_path, feature_name, feature_data):
    stat = open_csv_write(base_path + feature_name)

    # sort by value
    # feature_data = sorted(data[p], key=lambda x: (x[0]))
    # sort by artist
    # feature_data = sorted(data[p], key=lambda x: (x[2]))

    for record in feature_data:
        value = record[2]
        value_str = None

        if is_float(value):
            value_str = round(value, 3)

        elif is_tuple(value):
            if is_float(value[0]):
                value_str = ", ".join(map(lambda x: str(x), value))

            else:
                value_str = ""
                for k, element in enumerate(value):
                    value_str += ", ".join(map(lambda x: str(x), element))
                    if k < len(value) - 1: value_str += ", "

        stat.write("{0},{1},{2}\n".format(record[0], record[1], value_str))

    stat.close()


def refresh_base(check_base, mode, n_tracks):
    do_process(mode, basic_funcs.keys(), basic_funcs, check_base=check_base, n_tracks=n_tracks)


def rebuild_base(mode, n_tracks):
    refresh_base(False, mode, n_tracks)


def update_base(mode, n_tracks):
    refresh_base(True, mode, n_tracks)


def arg_run():
    # extractor.py
    # -e <feature>          calculate feature, rewrite feature file
    # -u                    update base (add new songs)
    # -r                    rebuild base (recalculate all features)

    parser = argparse.ArgumentParser(description="i <3 music! "
                                                 "this is feature extractor")
    parser.add_argument('-e', nargs=1)
    parser.add_argument('-n', nargs=1)
    parser.add_argument('-u', action='store_true')
    parser.add_argument('-r', action='store_true')
    parser.add_argument('-t', action='store_true')
    args = parser.parse_args()

    if sum(map(bool, [args.e, args.u, args.r])) > 1:
        print("Error: too many arguments")
        exit()

    mode = "favourite"
    if args.t is True:
        mode = "tag"

    n_tracks = None
    if args.n is not None:
        n_tracks = int(args.n[0])

    if args.e is not None:
        feature = args.e[0]
        if feature in basic_funcs.keys():
            do_process(mode, [args.e[0]], basic_funcs, check_base=False,
                       n_tracks=n_tracks)
        else:
            do_process(mode, [args.e[0]], experiment_funcs, check_base=False,
                       n_tracks=n_tracks)

    elif args.u is True:
        update_base(mode, n_tracks)

    elif args.r is True:
        rebuild_base(mode, n_tracks)

basic_funcs = {
            'bpm': e_bpm,
            'centroid': e_centroid,
            'volume': e_volume,
            'self-correlation': e_correlation,
            'zero_cross': e_zero_cross,
            'onset_strength': e_onset_strength,
            'onset_regular': e_onset_regular,
            'spectral_flux': e_flux,
            'spectral_contrast': e_contrast,
            'spectral_median': e_spec_median, # ?
            'spectral_std': e_spec_std, # ?
            'spectral_bandwidth': e_spec_bandwidth,
            'spectral_rolloff': e_spec_rolloff,
            'cepstral': e_cepstral
        }

experiment_funcs = {
            'chords': e_chords,
            'harmony': e_harmony
        }

feature_names = basic_funcs.keys()

if __name__ == "__main__":
    chdir(dirname(realpath(__file__)))

    if len(sys.argv) > 1:
        arg_run()
    else:
        pass