#!/Users/pavel/anaconda/bin/python
# !/usr/bin/env python

import numpy as np
from config import *
from pprint import pprint
import json
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import sys, argparse

features_names = ["bpm",
                  "centroid",
                  "onset_regular",
                  "onset_strength",
                  "self-correlation",
                  "spectral_contrast",
                  "spectral_flux",
                  "spectral_rolloff",
                  "spectral_bandwidth",
                  "volume",
                  "zero_cross"]


def load(fn, sort_needed=False):
    with open(fn, 'r') as file: content = file.readlines()

    content = content[1:]
    data = []
    for line in content:
        vec = line.split(",")
        data.append([float(vec[2]), vec[1].rstrip(), vec[0].rstrip()])
        #if len(vec[:-1]) == 2:
        #    data.append([float(vec[0]), vec[-1].rstrip()])
        #else:
        #    data.append([map(float, vec[:-1]), vec[-1].rstrip()])
    if sort_needed:
        return sorted(data, key=lambda x: (x[1]))
    else:
        return data
"""
def cosine(v1, v2):
    return np.dot(v1 / np.linalg.norm(v1),
                  v2 / np.linalg.norm(v2))

def vnorm(v):
    v = np.asarray(v)
    return 1.0 * v / np.sum(v)

def closest_scalars(target, data, nclosest = 5):
    idx = [line[1] for line in data].index(target)
    value = data[idx][0]

    vrange = [line[0] for line in data]
    wide = max(vrange) - min(vrange)

    for j in range(len(data)):
        data[j][0] = 1.0 * abs(data[j][0] - value) / wide

    #closest = sorted(data, key=lambda x: (x[0]))[1:nclosest + 1]
    closest = sorted(data, key=lambda x: (x[0]))
    return closest


def closest_vectors(target, data, nclosest = 5):
    for j in range(len(data)):
        data[j][0] = vnorm(data[j][0])

    idx = [line[-1] for line in data].index(target)
    vector = data[idx][0]

    for j in range(len(data)):
        data[j][0] = 1.0 - cosine(vector, data[j][0])

    #closest = sorted(data, key=lambda x: (-x[0]))[1:nclosest + 1]
    closest = sorted(data, key=lambda x: (x[0]))
    return closest
"""


def feature_normalise(feature, correct_nan=False):
    if correct_nan: feature[np.isnan(feature)] = 0
    n_feature = 1.0 * (feature - np.mean(feature)) / np.std(feature)
    return n_feature


def load_csv(fn):
    with open(basepath + fn, 'r') as file:
        content = file.readlines()
    content = content[1:]
    return map(lambda x: x.rstrip(), content)


def classifier(formula_file, out_file, play_file = None):
    props = json.load(open(formula_file, "r"))

    features_names = props["features"]
    feature_weights = props["weights"]

    assert(len(features_names) == len(feature_weights))

    n_features = len(features_names)
    data_bank = []
    for feature in features_names:
        data_bank.append(load("{0}{1}.csv".format(basepath, feature), True))

    song_idx = {}
    for i, song in enumerate(data_bank[0]):
        song_idx[i] = [song[1], song[2]]

    feature_bank = []
    for data in data_bank:
        feature_bank.append([song[0] for song in data])

    for k in range(n_features):
        feature_bank[k] = feature_weights[k] * feature_normalise(feature_bank[k])

    complex_feature = []
    n_songs = len(feature_bank[0])
    for k in range(n_songs):
        val = 0.0
        for feature in feature_bank:
            val += feature[k]
        complex_feature.append(val)

    file = open(out_file, "w+")
    file.write("sep=,\n")
    file.write("features,summary,song,artist\n")

    feature_base = []
    for j in range(n_songs):
        values_list = [str(round(feature[j], 3)) for feature in feature_bank]
        feature_base.append([round(complex_feature[j], 3),
                            [song_idx[j][0], song_idx[j][1]],
                             values_list])

    feature_base = sorted(feature_base, key=lambda x: (x[0]), reverse=True)

    cut_flag = False
    cut_artists = []
    if play_file is not None:
        cut_artists = json.load(open(play_file, "r"))["negative"]

    for j in range(n_songs):
        if feature_base[j][1][1] in cut_artists and \
                        len(cut_artists) > 0 \
                        and not cut_flag:
            cut_flag = True
            file.write("*,*,*,*\n")

        file.write("{},".format("; ".join(feature_base[j][2])))
        file.write("{},{},{}\n".format(feature_base[j][0],
                                       feature_base[j][1][0],
                                       feature_base[j][1][1]))


def print_dict(d):
    for f in sorted(d, key=d.get, reverse=True):
        print str(round(d[f], 4)) + "\t\t" + str(f)


def features_esimate(playlist, n_validation = 10, classifier_type = "boosting"):
    props = json.load(open(playlist, "r"))

    data_bank = []
    # format: [value, song_name, artist]
    for feature_name in features_names:
        data_bank.append(load(basepath + feature_name + ".csv", True))

    feature_bank = []
    for data in data_bank:
        feature_bank.append([song[0] for song in data])

    for p in range(len(feature_bank)):
        feature_bank[p] = feature_normalise(feature_bank[p])

    feature_bank = np.array(feature_bank).T
    song_idx = {}
    artist_idx = {}

    for k, song in enumerate(data_bank[0]):
        song_name = song[1]
        artist = song[2]
        song_idx[song_name] = k
        if artist not in artist_idx.keys():
            artist_idx[artist] = [k]
        else:
            artist_idx[artist].append(k)

    all_songs = np.array(data_bank[0]).T[1].tolist()
    pos_songs = []
    neg_songs = []

    if props["type"] == 'song':
        pos_songs = props["positive"]

        neg_songs = []
        if len(props["negative"]) > 0:
            neg_songs = props["negative"]
        else:
            for k in range(len(all_songs)):
                if all_songs[k] not in pos_songs:
                    neg_songs.append(all_songs[k])

    elif props["type"] == 'artist':
        for song in data_bank[0]:
            if song[2] in props["positive"]:
                pos_songs.append(song[1])
            if song[2] in props["negative"]:
                neg_songs.append(song[1])

    flags = np.array([1 if song in props["positive"]
                      else 0 for song in all_songs])
    s_err = 0.0
    i_val = 0.0
    pos_idx = [all_songs.index(song) for song in pos_songs]
    neg_idx = [all_songs.index(song) for song in neg_songs]

    for n in range(n_validation):
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)

        train_pos_idx = pos_idx[:len(pos_idx) / 2]
        test_pos_idx = pos_idx[len(pos_idx) / 2:]

        train_neg_idx = neg_idx[:len(neg_idx) / 2]
        test_neg_idx = neg_idx[len(neg_idx) / 2:]

        train_idx = train_pos_idx + train_neg_idx
        test_idx = test_pos_idx + test_neg_idx

        clf = None
        if classifier_type == "boosting":
            clf = GradientBoostingClassifier(n_estimators=11,
                                             learning_rate=0.2,
                                             max_depth=1,
                                             random_state=1).fit(feature_bank[train_idx],
                                                                              flags[train_idx])
        elif classifier_type == "forest":
            clf = RandomForestClassifier(n_estimators=11,
                                         random_state=1,
                                         max_features=5,
                                         max_depth=5).fit(feature_bank[train_idx],
                                                        flags[train_idx])
        i_val += clf.feature_importances_
        s_err += clf.score(feature_bank[test_idx], flags[test_idx])

    print "test accuracy", s_err / n_validation
    print "features importance"
    print_dict(dict(zip(features_names, i_val / n_validation)))


def features_probe(playlist, classifier_type = "boosting"):
    props = json.load(open(playlist, "r"))

    data_bank = []
    # format: [value, song_name, artist]
    for feature_name in features_names:
        data_bank.append(load(basepath + feature_name + ".csv", True))

    feature_bank = []
    for data in data_bank:
        feature_bank.append([song[0] for song in data])

    for p in range(len(feature_bank)):
        feature_bank[p] = feature_normalise(feature_bank[p])

    feature_bank = np.array(feature_bank).T
    song_idx = {}
    artist_idx = {}

    for k, song in enumerate(data_bank[0]):
        song_name = song[1]
        artist = song[2]
        song_idx[song_name] = k
        if artist not in artist_idx.keys():
            artist_idx[artist] = [k]
        else:
            artist_idx[artist].append(k)

    all_songs = np.array(data_bank[0]).T[1].tolist()
    pos_songs = []
    neg_songs = []

    if props["type"] == 'song':
        pos_songs = props["positive"]

        neg_songs = []
        if len(props["negative"]) > 0:
            neg_songs = props["negative"]
        else:
            for k in range(len(all_songs)):
                if all_songs[k] not in pos_songs:
                    neg_songs.append(all_songs[k])

    elif props["type"] == 'artist':
        for song in data_bank[0]:
            if song[2] in props["positive"]:
                pos_songs.append(song[1])
            if song[2] in props["negative"]:
                neg_songs.append(song[1])

    flags = np.array([1 if song in props["positive"]
                      else 0 for song in all_songs])

    pos_idx = [all_songs.index(song) for song in pos_songs]
    neg_idx = [all_songs.index(song) for song in neg_songs]
    train_idx = pos_idx + neg_idx

    clf = None
    if classifier_type == "boosting":
        clf = GradientBoostingClassifier(n_estimators=11,
                                         learning_rate=0.2,
                                         max_depth=1,
                                         random_state=1).fit(feature_bank[train_idx],
                                                                          flags[train_idx])
    elif classifier_type == "forest":
        clf = RandomForestClassifier(n_estimators=11,
                                     random_state=1,
                                     max_features=5,
                                     max_depth=5).fit(feature_bank[train_idx],
                                                    flags[train_idx])
    d = {}
    for k in range(len(all_songs)):
        d[all_songs[k]] = clf.predict_proba(feature_bank[k])[0][1]
    print_dict(d)


def closest_songs(song_name, top_n = 20):
    data_bank = []
    # format: [value, song_name, artist]
    for feature_name in features_names:
        data_bank.append(load(basepath + feature_name + ".csv", True))

    feature_bank = []
    names_bank = []
    for data in data_bank:
        feature_bank.append([song[0] for song in data])
    names_bank=[song[1] for song in data]

    for p in range(len(feature_bank)):
        feature_bank[p] = feature_normalise(feature_bank[p])

    feature_bank = np.array(feature_bank).T
    song_idx = {}
    artist_idx = {}

    for k, song in enumerate(data_bank[0]):
        song_name = song[1]
        artist = song[2]
        song_idx[song_name] = k
        if artist not in artist_idx.keys():
            artist_idx[artist] = [k]
        else:
            artist_idx[artist].append(k)

    target_idx = song_idx[song_name]
    target_feature = feature_bank[target_idx]
    complex_feature = []
    n_songs = len(names_bank)

    for k in range(n_songs):
        complex_feature.append(np.sum((np.array(feature_bank[k]) -
                np.array(target_feature))**2))

    result = [[names_bank[k],
               complex_feature[k]] for k in range(n_songs)]

    pprint(sorted(result, key=lambda x: (x[1]))[:top_n])


def arg_run():
    # classifier.py
    # -c <file_formula>.json   run classifier with formula, JSON loaded from config.formula_path,
    #                           features loaded from config.base_path and composition saved to config.complex_path
    # -e <file_playlist>.json  estimate feature importances by playlist, JSON loaded from config.hand_class_path
    # -p <file_playlist>.json  estimate class probability by playlist, JSON loaded from config.hand_class_path

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-W', nargs=1)
    parser.add_argument('-c', nargs=1)
    parser.add_argument('-e', nargs=1)
    parser.add_argument('-p', nargs=1)
    args = parser.parse_args()

    if sum(map(bool, [args.c, args.e, args.p])) > 1:
        print "Error: too many arguments"
        exit()

    if args.c is not None:
        classifier(formula_path + args.c[0] + ".json",
                   complex_path + args.c[0] + ".csv")

    elif args.e is not None:
        features_esimate(hand_class_path + args.e[0] + ".json",
                         classifier_type="boosting")

    elif args.p is not None:
        features_probe(hand_class_path + args.p[0] + ".json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg_run()
    else:
        pass
        #classifier(formula_path + "dance.json", complex_path + "dance.csv")
        #classifier(formula_path + "morning.json",
        #           hand_class_path + "morning.json",
        #           complex_path + "morning.csv")
        #classifier(formula_path + "evening.json", complex_path + "morning.csv")
        #classifier(formula_path + "agressive.json", complex_path + "agressive.csv")
        #classifier(formula_path + "happy.json", complex_path + "happy.csv")

        #features_esimate(classifier_type="boosting")
        #features_esimate(classifier_type="forest")
        #features_esimate(hand_class_path+"dance.json")
        #features_probe("../data/forest.json")

        #closest_songs("Bohemian_Rhapsody")

