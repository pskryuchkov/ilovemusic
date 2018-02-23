from sklearn.ensemble import GradientBoostingRegressor
from os.path import realpath, dirname
from numpy import *
import pickle
import csv

features_names = ["artist_song", "bpm_median",  # #
                        "bpm_std",  # # # #
                        "centroid_mean",  # # #
                        "centroid_std",  # # # # #
                        "onset_regular",  # # # #
                        "onset_strength_mean",  # #
                        "onset_strength_std",  # # #
                        "self-correlation",  # # # #
                        "spectral_contrast_mean",  #
                        "spectral_contrast_std",  #
                        "spectral_flux_mean",
                        "spectral_flux_std",  # # #
                        "spectral_rolloff_mean",  #
                        "spectral_rolloff_std",  # # # #
                        "spectral_bandwidth_mean",  # # # #
                        "spectral_bandwidth_std",  # # # #
                        "volume_mean",  # # # #
                        "volume_std",  #
                        "zero_cross",  #
                        "cepstral_mean1",  # # # # #
                        "cepstral_mean2",  # # #
                        "cepstral_mean3",  # # # #
                        "cepstral_mean4",  # # # # # #
                        "cepstral_mean5",  # # # #
                        "cepstral_mean6",
                        "cepstral_mean7",  # # #
                        "cepstral_mean8",  # # # #
                        "cepstral_mean9",
                        "cepstral_mean10",  # # # # #
                        "cepstral_std1",  # # # #
                        "cepstral_std2",  # # # #
                        "cepstral_std3",  #
                        "cepstral_std4",  # # #
                        "cepstral_std5",  #
                        "cepstral_std6",  # # #
                        "cepstral_std7",  # # #
                        "cepstral_std8",  #
                        "cepstral_std9",  #
                        "cepstral_std10"] # # #


spotify_features = ["popularity",
                    "danceability",
                    "energy",
                    "speechiness",
                    "instrumentalness",
                    "liveness",
                    "valence",
                    "tempo"
                    ]


def load_features(ff):
    physical = []

    with open(ff, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')

        for j, row in enumerate(reader):
            features = list(map(lambda x: row[x], features_names))
            physical.append(features)

    return array(physical)


def load_target(ff, h_name):
    z = []

    with open(ff, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')

        for j, row in enumerate(reader):
            artist_song = row["artist"] + " " + row["song"]
            z.append([artist_song, row[h_name]])

    return array(z)


def find_pairs(names1, names2):
    ns1 = [set(x.split()) for x in names1]
    ns2 = [set(x.split()) for x in names2]

    matches = []
    for j, s1 in enumerate(ns1):
        for k, s2 in enumerate(ns2):
            candidates = []
            if len(s1 - s2) == 0:
                candidates.append([j, k])

            if len(candidates) == 1:
                matches += candidates

    return matches


def match(list1, list2, pairs):
    s_list1, s_list2 = [], []

    for pair in pairs:
        idx1, idx2 = pair[0], pair[1]
        assert(idx1 < len(list1))
        assert(idx2 < len(list2))

        s_list1.append(list1[idx1])
        s_list2.append(list2[idx2])

    return array(s_list1), array(s_list2)


def matrix_to_float(ls):
    new_ls = []
    for line in ls:
        new_ls.append(list(map(float, line)))

    return array(new_ls)

origin = dirname(realpath(__file__)) + "/"

features = load_features(origin + "physical_features.csv")
target = load_target(origin + "spotify_features.csv", "popularity")

features = array(list(filter(lambda x: not 'nan' in x, features)))

physical_names = [x[0].replace("_", " ").lower().strip() for x in features]
spotify_names = [x[0].lower() for x in target]

features = matrix_to_float(features[:, 1:])

pairs = find_pairs(spotify_names, physical_names)

# debug
# import pickle
# pickle.dump(pairs, open("pairs", "wb"))

# pairs = pickle.load(open("pairs", "rb"))

for hf in spotify_features:

    target = load_target(origin + "spotify_features.csv", hf)

    target = array(list(map(float, target[:, 1])))

    s_target, s_features = match(target, features, pairs)

    s_features = s_features.reshape(s_features.shape[0], -1)

    regressor = GradientBoostingRegressor()

    regressor.fit(s_features, s_target)

    print("{} (r^2={:.2}, D={:.3})".format(hf, regressor.score(s_features, s_target), \
          sqrt(mean((s_target - regressor.predict(s_features)) ** 2))))

    pickle.dump(regressor, open(origin + "../../data/regressors/{}".format(hf), 'wb'))



