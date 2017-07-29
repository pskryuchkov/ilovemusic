favourite_path = "../music/favourite"
tag_path = "../music/tags"

fav_songs_stat = "../data/features/favourite/"
tag_songs_stat = "../data/features/tags/"

classifiers_bank = "../data/classifiers/{}.clf"

music_ext = [".mp3", ".wav"]

durs = {
            'bpm': 60,
            'self-correlation': 20,
            'zero_cross': 30,
            'onset_strength': 60,
            'onset_regular': 40,
            'centroid': 60,
            'volume': 60,
            'spectral_flux': 40,
            'spectral_contrast': 40,
            'spectral_std': 40,
            'spectral_median': 40,
            'spectral_bandwidth': 40,
            'spectral_rolloff': 40,
            'chords': 40,
            'cepstral': 30
        }

features_files = ["bpm",
                        "centroid",
                        "onset_regular",
                        "onset_strength",
                        "self-correlation",
                        "spectral_contrast",
                        "spectral_flux",
                        "spectral_rolloff",
                        "spectral_bandwidth",
                        "volume",
                        "zero_cross",
                        "cepstral"]

features_names = ["bpm_median",  # #
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