#!/Users/pavel/anaconda/bin/python
# !/usr/bin/env python

import numpy as np
from config import *
from pprint import pprint
import json
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import sys, argparse
import pickle
from os.path import isfile
from sklearn.feature_selection import RFE

# deprecated
"""
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
"""
"""                                               # alphabetically sorted!
features_names_dict = {
                        "bpm":                  { "median", "std" },
                        "centroid":             { "mean", "std" },
                        "onset_regular":        { },
                        "onset_strength":       { "mean", "std" },
                        "self-correlation":     { },
                        "spectral_contrast":    { "mean", "std" },
                        "spectral_flux":        { "mean", "std" },
                        "spectral_rolloff":     { "mean", "std" },
                        "spectral_bandwidth":   { "mean", "std" },
                        "volume":               { "mean", "std" },
                        "zero_cross":           { },
                        "cepstral":             { "mean0", "mean1", "mean2", "mean3", "mean4",
                                                    "mean5", "mean6", "mean7", "mean8", "mean9",
                                                    "std0", "std1", "std2", "std3", "std4",
                                                    "std5","std6", "std7", "std8", "std9" } }
"""



# deprecated
"""
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

# deprecated
"""
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
"""

# deprecated
"""
def print_dict(d):
    for f in sorted(d, key=d.get, reverse=True):
        print str(round(d[f], 4)) + "\t\t" + str(f)
"""

# deprecated
"""
def features_esimate2(playlist, n_validation = 10, classifier_type = "boosting"):
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

    #for n in range(n_validation):
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    train_pos_idx = pos_idx[:len(pos_idx) / 2]
    test_pos_idx = pos_idx[len(pos_idx) / 2:]

    train_neg_idx = neg_idx[:len(neg_idx) / 2]
    test_neg_idx = neg_idx[len(neg_idx) / 2:]

    train_idx = train_pos_idx + train_neg_idx
    test_idx = test_pos_idx + test_neg_idx

    #clf = None

    from sklearn.feature_selection import RFE
    b = GradientBoostingClassifier(n_estimators=11,
                                     learning_rate=0.2,
                                     max_depth=1,
                                     random_state=1)
    rfe = RFE(b, 5)
    res = rfe.fit(feature_bank[train_idx], flags[train_idx])
    print res.support_
    print res.get_support()
    print res.ranking_
    for k, relevent_flag in enumerate(res.support_):
        if relevent_flag:
            print features_names[k]
    #    i_val += clf.feature_importances_
    #    s_err += clf.score(feature_bank[test_idx], flags[test_idx])
    #
    #print "test accuracy", s_err / n_validation
    #print "features importance"
    #print_dict(dict(zip(features_names, i_val / n_validation)))
"""

# fix errors
"""
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
"""

# deprecated
"""
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
"""
# rewrite
"""
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
"""

def complex_load(fn):
    content = open(fn, 'r').readlines()
    content = content[1:] # remove first line

    return [line.split(",") for line in content]

"""
def get_features_names():
    features_names_keys = sorted(features_names_dict.keys())
    print features_names_keys
    f = []
    for key in features_names_keys:
        e_key = []
        if len(features_names_dict[key]) == 0:
            e_key.append(key)
        else:
            for subkey in features_names_dict[key]:
                e_key.append(key + "_" + subkey)

        f.append(sorted(e_key))
    return unzip(f)
"""

def b_feature_bank(basepath, normalize=False):
    data_bank = []

    for feature_name in features_files:
        data_bank.append(complex_load(basepath + feature_name + ".csv"))

    artists = [line[0] for line in data_bank[0]]
    songs = [line[1] for line in data_bank[0]]

    raw_feature_bank = []
    for j in range(len(features_files)):
        # skip song and artist
        raw_feature_bank.append([feature[2:] for feature in data_bank[j]])

    # kolonki - fichi, stroki - pesni
    raw_feature_bank = np.array(raw_feature_bank).T
    feature_bank = np.array([unzip([map(lambda x: float(x), x)
                                for x in song_features.tolist()])
                                for song_features in raw_feature_bank])

    #for p, x in enumerate(feature_bank[0]):
    #    print x, features_names_long[p]

    if normalize:
        for p in range(feature_bank.shape[1]):
            feature_bank[:, p] = feature_normalise(feature_bank[:, p])

    return artists, songs, feature_bank

"""
def build_feature_bank(basepath, normalize=True):
    data_bank = []
    feature_bank = []

    # format: [value, song_name, artist]
    for feature_name in features_names:
        data_bank.append(load(basepath + feature_name + ".csv", True))

    for data in data_bank:
        feature_bank.append([song[0] for song in data])

    if normalize:
        for p in range(len(feature_bank)):
            feature_bank[p] = feature_normalise(feature_bank[p])

    return np.array(feature_bank).T


def get_songs_names(basepath):
    data_bank = []
    feature_bank = []

    # format: [value, song_name, artist]
    for feature_name in features_names:
        data_bank.append(load(basepath + feature_name + ".csv", True))

    for data in data_bank:
        feature_bank.append([song[1] for song in data])

    return feature_bank[0]


def get_artist_names(basepath):
    data_bank = []
    feature_bank = []

    # format: [value, song_name, artist]
    for feature_name in features_names:
        data_bank.append(load(basepath + feature_name + ".csv", True))

    for data in data_bank:
        feature_bank.append([song[2] for song in data])

    return feature_bank[0]
"""


def get_top_bands(top, rank_func='mean'):
    top_pos = [[222-j, top[j][0]] for j in range(len(top))]
    points_dict = {}
    for record in top_pos:
        if record[1] in points_dict.keys():
            points_dict[record[1]].append(record[0])
        else:
            points_dict[record[1]] = [record[0]]

    points = []
    if rank_func == 'mean':
        for key in points_dict.keys():
            points_dict[key] = round(np.mean(points_dict[key]), 2)

        for key in points_dict.keys():
            points.append([key, points_dict[key]])
    elif rank_func == 'sum':
        for key in points_dict.keys():
            points_dict[key] = round(np.sum(points_dict[key]), 2)
        for key in points_dict.keys():
            points.append([key, points_dict[key]])

    return sorted(points, key=lambda x: x[1], reverse=True)


def mark_tag(target_class, tags_bank):
    flags = []
    positive_idx = []
    negative_idx = []

    for i in range(len(tags_bank)):
        if tags_bank[i] == target_class:
            flags.append(1)
            positive_idx.append(i)
        else:
            flags.append(0)
            negative_idx.append(i)

    return np.array(flags), \
           positive_idx, \
           negative_idx


def mark_artist(target_artist, artist_bank):
    flags = []
    for i in range(len(artist_bank)):
        if artist_bank[i] == target_artist:
            flags.append(1)
        else:
            flags.append(0)
    return flags


def binary_classificator(features, classes, train_part=0.6):
    n_songs = len(classes)
    n_train = int(train_part * n_songs)

    pos_idx = [j for j in range(len(classes)) if classes[j] == 1]
    neg_idx = [j for j in range(len(classes)) if classes[j] == 0]

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    train_features = features[pos_idx[:n_train] + neg_idx[:n_train],:]
    train_classes = classes[pos_idx[:n_train] + neg_idx[:n_train]]

    if train_part < 1.0:
        test_features = features[pos_idx[n_train:] + neg_idx[n_train:], :]
        test_classes = classes[pos_idx[n_train:] + neg_idx[n_train:]]

    clf = GradientBoostingClassifier(n_estimators=features.shape[1],
                                     learning_rate=0.2,
                                     max_depth=2,
                                     random_state=1).fit(train_features, train_classes)

    print "train score: ", clf.score(train_features, train_classes)
    if train_part < 1.0:
        print "test score:  ", clf.score(test_features, test_classes)

    return clf


def relevant_music(artists, songs, class_results):
    top_songs = sorted([[artists[i], songs[i],
                    class_results[i][1]] for i in range(len(class_results))],
                  key=lambda x: (-x[2]))

    top_bands = get_top_bands(top_songs)

    bands_points = np.array([record[1] for record in top_bands])

    der = bands_points[:-1] - bands_points[1:]
    print "der",der
    n_relevant = np.argmax(der[:der.size / 2])
    print n_relevant, der[np.argmax(der[:der.size / 2])]

    #relevant_bands = [x[0] for x in top_bands[:7]]
    #relevant_songs = [x for x in top_songs if x[0] in relevant_bands]

    pprint(top_bands[:n_relevant+1])
    return top_bands, top_songs


def save_feature_info(tag, features_idx):
    fn = "top_features.json"

    if isfile(fn):
        i_dict = dict(json.load(open(fn, "r")))
    else:
        i_dict = {}

    i_dict[tag] = [features_names[idx] for idx in features_idx]

    json.dump(i_dict, open(fn, "w+"), indent=1)


def class_proba():
    target_class = "melancholia"
    print "class:", target_class
    n_features = 14

    tags, _, tag_bank = b_feature_bank(tag_features_bank)
    fav_artists, fav_songs, fav_bank = b_feature_bank(fav_features_bank)

    flags, pos_idx, neg_idx = mark_tag(target_class, tags)

    features_idx = features_choose(tag_bank, flags, target_class, topn=n_features)

    save_feature_info(target_class, features_idx)

    tag_bank = tag_bank[:, features_idx]
    fav_bank = fav_bank[:, features_idx]

    classifier = binary_classificator(tag_bank, flags)

    c_results = classifier.predict_proba(fav_bank)

    top_bands, top_songs = relevant_music(fav_artists, fav_songs, c_results)


    import matplotlib.pyplot as plt
    plt.plot([record[1] for record in top_bands])
    plt.plot([record[1] for record in top_bands] ,"x")
    pprint(top_bands)
    pprint(top_songs)
    plt.show()
    #with open('../data/classifiers/{}.clf'.format(target_class), 'wb') as f:
    #    pickle.dump(classifier, f)


def features_choose(tag_bank, flags, target_class, topn=6):

    clf = GradientBoostingClassifier(n_estimators=39,
                                     learning_rate=0.2,
                                     max_depth=1,
                                     random_state=1)

    rfe = RFE(clf, 1)
    res = rfe.fit(tag_bank, flags)

    rank = sorted(zip(range(len(features_names)), res.ranking_), key=lambda x: (x[1]))
    #pprint(sorted(zip(features_names_long, res.ranking_), key=lambda x: (x[1])))
    #pprint(sorted(zip(features_names, res.ranking_), key=lambda x: (x[1]))[:topn])
    rank = rank[:topn]

    return [pos[0] for pos in rank]


def cosine(u, v):
    u = np.array(u)
    v = np.array(v)
    return np.sum(u*v / (np.linalg.norm(u) * np.linalg.norm(v)))

# merge with closest songs
"""
def closest_songs_by_metric():
    target_class1 = "sad"

    features_bank = build_feature_bank("../data/features/basic/tags/", False)

    flags = mark_by_class(target_class1)
    tag_bank = np.array([features_bank[j] for j in range(len(flags)) if flags[j] == 1])

    favourite_bank = build_feature_bank("../data/features/basic/albums/", False)
    favourite_songs = get_songs_names("../data/features/basic/albums/")
    favourite_artists = get_artist_names("../data/features/basic/albums/")

    features_idx = rfe_estimate(target_class1, topn=6)
    tag_bank = tag_bank[:, features_idx]
    favourite_bank = favourite_bank[:, features_idx]

    tag_centroid = np.mean(tag_bank, axis=0)
    tag_std = np.std(tag_bank, axis=0)
    favourite_bank = np.copy(1.0 * (favourite_bank - tag_centroid) / tag_std)

    # cosine
    #dists = []
    #pprint()

    # euclid
    dists = np.linalg.norm(favourite_bank, axis=1)
    top = sorted([[favourite_artists[i],
                    favourite_songs[i],
                    dists[i]] for i in range(len(dists))],
                  key=lambda x: (x[2]))
    pprint(top_bands(top))
"""


def artist_stat():
    # incorrect ???
    target_artist = "Depeche_Mode"
    print "artist:", target_artist

    top_features = json.load(open("top_features.json", "r"))

    artists, songs, features_bank = b_feature_bank(fav_features_bank)
    artist_flags = mark_artist(target_artist, artists)
    artist_bank = np.array([features_bank[j] for j in range(len(artist_flags))
                            if artist_flags[j] == 1])

    tags, _, features_bank = b_feature_bank(tag_features_bank)
    tags = top_features.keys()

    artist_stat = []
    for tag in tags:
        features_idx = [j for j in range(len(features_names))
                        if features_names[j] in top_features[tag]]

        with open(classifiers_bank.format(tag), 'rb') as f:
            clf = pickle.load(f)

        vals = []
        for song_features in artist_bank:
            n_song_features = np.array(song_features)[features_idx]
            vals.append(clf.predict_proba([n_song_features])[0][1])

        artist_stat.append([tag, np.mean(vals)])

    pprint(sorted(artist_stat, key=lambda x: x[1], reverse=True))

# fix this govnokod
def unzip(a):
    b = []
    for x in a: b += x
    return b


# test me!
def feature_raiting(target_class, n_validation = 10):
    tags, _, tag_bank = b_feature_bank(tag_features_bank)
    flags, _, _ = mark_tag(target_class, tags)

    s_err = 0.0
    i_val = 0.0
    for n in range(n_validation):
        clf = binary_classificator()
        i_val += clf.feature_importances_
        s_err += clf.score(tag_bank, flags)

    print "test accuracy", s_err / n_validation
    print "features importance"
    #print_dict(dict(zip(features_names, i_val / n_validation)))

# fix errors
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
    #if len(sys.argv) > 1:
    #    arg_run()
    #else:
    #    pass

    # deprecated
    """
    #features_esimate2(hand_class_path + "dance" + ".json",
    #                 classifier_type="boosting")

    #features_esimate(hand_class_path + "dance" + ".json",
    #                 classifier_type="boosting")

    #classifier(formula_path + "dance.json", complex_path + "dance.csv")
    #classifier(formula_path + "morning.json",
    #           hand_class_path + "morning.json",
    #           complex_path + "morning.csv")
    #classifier(formula_path + "evening.json", complex_path + "morning.csv")
    #classifier(formula_path + "agressive.json", complex_path + "agressive.csv")
    #classifier(formula_path + "happy.json", complex_path + "happy.csv")

    #features_esimate(classifier_type="boosting")
    #features_esimate(classifier_type="forest")
    #features_esimate2(hand_class_path+"dance.json")
    #features_probe("../data/forest.json") !!!

    #closest_songs("The_Black_Dahlia_Murder_Climatic_Degradation")
    """
    #artist_stat()
    class_proba()