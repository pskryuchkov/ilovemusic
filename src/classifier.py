#!/Users/pavel/anaconda/bin/python
# !/usr/bin/env python

import numpy as np
from config import *
from pprint import pprint
import json
from sklearn.ensemble import GradientBoostingClassifier, \
    RandomForestClassifier
import sys
import argparse
import pickle
from os.path import isfile
from sklearn.feature_selection import RFE


def load_csv(fn, sep=","):
    content = open(fn, 'r').readlines()
    content = content[1:]  # remove first line
    return [line.split(sep) for line in content]


def feature_normalise(values, correct_nan=False):
    if correct_nan:
        values[np.isnan(values)] = 0
    n_feature = 1.0 * (values - np.mean(values)) / np.std(values)
    return n_feature


def load_features_bank(basepath, normalize=False):
    data_bank = []

    for feature_name in features_files:
        data_bank.append(load_csv(basepath + feature_name + ".csv"))

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

    if normalize:
        for p in range(feature_bank.shape[1]):
            feature_bank[:, p] = feature_normalise(feature_bank[:, p])

    return artists, songs, feature_bank


def bands_raiting(top, rank_func='mean'):
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


def binary_classifier(features, classes, train_part=0.6, debug_info=True):
    n_songs = len(classes)
    n_train = int(train_part * n_songs)

    pos_idx = [j for j in range(len(classes)) if classes[j] == 1]
    neg_idx = [j for j in range(len(classes)) if classes[j] == 0]

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    train_features = features[pos_idx[:n_train] + neg_idx[:n_train], :]
    train_classes = classes[pos_idx[:n_train] + neg_idx[:n_train]]

    test_classes = None
    test_features = None
    if train_part < 1.0:
        test_features = features[pos_idx[n_train:] + neg_idx[n_train:], :]
        test_classes = classes[pos_idx[n_train:] + neg_idx[n_train:]]

    # choose best classifier!
    clf = GradientBoostingClassifier(n_estimators=features.shape[1],
                                     learning_rate=0.2,
                                     max_depth=3,
                                     random_state=1).fit(train_features, train_classes)

    if debug_info:
        print "train score: ", round3(clf.score(train_features, train_classes))
        if train_part < 1.0:
            print "test score:  ", round3(clf.score(test_features, test_classes))

    return clf


def top(artists, songs, class_results):
    min_artists = 3

    top_songs = sorted([[artists[i], songs[i],
                    round3(class_results[i][1])] for i in range(len(class_results))],
                  key=lambda x: x[2], reverse=True)

    top_bands = bands_raiting(top_songs)

    bands_points = np.array([record[1] for record in top_bands])

    der = bands_points[min_artists-1:-1] - bands_points[min_artists:]
    n_relevant = min_artists - 1 + np.argmax(der[:int(0.4 * der.size)])
    top_bands = top_bands[:n_relevant+1]

    max_prob = top_songs[0][2]
    top_songs = [record for record in top_songs
                if record[2] > max_prob / 2.]

    return top_bands, top_songs


def save_features_names(tag, features_idx):
    fn = "top_features.json"

    if isfile(fn):
        i_dict = dict(json.load(open(fn, "r")))
    else:
        i_dict = {}

    i_dict[tag] = [features_names[idx] for idx in features_idx]

    json.dump(i_dict, open(fn, "w+"), indent=1)


def save_classifier(clf_name, clf):
    with open('../data/classifiers/{}.clf'.format(clf_name), 'wb') as f:
        pickle.dump(clf, f)


def draw_array(data):
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.show()


def class_relevant(target_class, n_features=12):
    print "class:", target_class

    tags, _, tag_bank = load_features_bank(tag_features_bank)
    fav_artists, fav_songs, fav_bank = load_features_bank(fav_features_bank)

    flags, pos_idx, neg_idx = mark_tag(target_class, tags)

    features_idx = sorted(features_choose(tag_bank, flags, topn=n_features))

    save_features_names(target_class, features_idx)

    tag_bank = tag_bank[:, features_idx]
    fav_bank = fav_bank[:, features_idx]

    classifier = binary_classifier(tag_bank, flags, 1.0)

    c_results = classifier.predict_proba(fav_bank)

    top_bands, top_songs = top(fav_artists, fav_songs, c_results)

    print "* top bands *"
    pprint(top_bands)
    print "* top songs *"
    pprint(top_songs)

    save_classifier(target_class, classifier)

    #draw_array([record[1] for record in top_bands])


def features_choose(tag_bank, flags, topn=6, debug_info=False):
    clf = GradientBoostingClassifier(n_estimators=39,
                                     learning_rate=0.2,
                                     max_depth=3,
                                     random_state=1)
    rfe = RFE(clf, 1)
    res = rfe.fit(tag_bank, flags)

    top_idx_rank = sorted(zip(range(len(features_names)), res.ranking_), key=lambda x: (x[1]))[:topn]
    top_names_rank = sorted(zip(features_names, res.ranking_), key=lambda x: (x[1]))[:topn]

    top_idx = [pos[0] for pos in top_idx_rank]
    top_names = [line[0] for line in top_names_rank]

    if debug_info: pprint(top_names)

    return top_idx


def cosine(u, v):
    return np.sum(np.array(u) * np.array(v) /
                  (np.linalg.norm(u) * np.linalg.norm(v)))


def closest_songs(song, features, topn=20):
    fav_artists, fav_songs, fav_bank = load_features_bank(fav_features_bank)
    features_idx = [features_names.index(feature) for feature in features]

    fav_bank = fav_bank[:, features_idx]
    fav_song = fav_bank[fav_songs.index(song),:]

    dists = []
    for song in fav_bank:
        dists.append(cosine(fav_song, song))

    pprint(sorted([[fav_songs[i], dists[i]]
                   for i in range(len(fav_artists))], key=lambda x: x[1], reverse=True)[:topn])


def song_disp(fav_artist, topn=10):
    print "artist:", fav_artist
    fav_artists, fav_songs, fav_bank = load_features_bank(fav_features_bank)
    artist_idx = [j for j, artist in enumerate(fav_artists) if artist == fav_artist]

    fav_bank = fav_bank[artist_idx, :]
    for j in range(fav_bank.shape[1]):
        fav_bank[:, j] = feature_normalise(fav_bank[:, j])

    features_mean = np.std(fav_bank, axis=1)
    pprint(sorted([[features_names[j], features_mean[j]]
                   for j in range(features_mean.shape[0])], key=lambda x: x[1])[:topn])


def round3(val):
    return round(val, 3)


def artist_stat(target_artist):
    print "artist:", target_artist

    top_features = json.load(open("top_features.json", "r"))

    artists, songs, features_bank = load_features_bank(fav_features_bank)
    artist_flags = mark_artist(target_artist, artists)
    artist_bank = np.array([features_bank[j] for j in range(len(artist_flags))
                            if artist_flags[j] == 1])

    tags = top_features.keys()

    stat = []
    for tag in tags:
        features_idx = [j for j in range(len(features_names))
                        if features_names[j] in top_features[tag]]

        with open(classifiers_bank.format(tag), 'rb') as f:
            clf = pickle.load(f)

        vals = []
        for song_features in artist_bank:
            n_song_features = np.array(song_features)[features_idx]
            vals.append(clf.predict_proba([n_song_features])[0][1])

        stat.append([tag, round3(np.mean(vals)), round3(np.std(vals))])

    pprint(sorted(stat, key=lambda x: x[1], reverse=True))


# fix this govnokod
def unzip(a):
    b = []
    for x in a: b += x
    return b


# test me!
def features_raiting(target_class, n_validation = 10, topn=10):
    print "class:", target_class
    tags, _, tag_bank = load_features_bank(tag_features_bank)
    flags, _, _ = mark_tag(target_class, tags)

    s_err = 0.0
    i_val = 0.0
    for n in range(n_validation):
        clf = binary_classifier(tag_bank, flags, 1.0, False)
        i_val += clf.feature_importances_
        s_err += clf.score(tag_bank, flags)

    print "train score:", round3(s_err / n_validation)

    pprint(sorted([[features_names[j], round3(i_val[j] / n_validation)]
           for j in range(len(features_names))],
                  key=lambda x: x[1], reverse=True)[:topn])
    #pprint(dict(zip(features_names, i_val / n_validation)))

# fix errors
def arg_run():
    # classifier.py

    parser = argparse.ArgumentParser(description="Flip a switch by setting a flag")
    parser.add_argument('-W', nargs=1)
    parser.add_argument('-t', nargs=1)
    parser.add_argument('-a', nargs=1)
    parser.add_argument('-p', nargs=1)
    parser.add_argument('-f', nargs=1)
    args = parser.parse_args()

    if sum(map(bool, [args.t, args.a, args.f])) > 1:
        print "Error: too many arguments"
        exit()

    if args.t is not None:
        # tag
        class_relevant(args.t[0])

    elif args.a is not None:
        # artist
        artist_stat(args.a[0])

    elif args.p is not None:
        # artist
        song_disp(args.p[0])

    elif args.f is not None:
        # tag
        features_raiting(args.f[0])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg_run()
    else:
        pass

    #closest_songs("06_Sunset", ["bpm_median", "bpm_std", "centroid_mean", "centroid_std"])