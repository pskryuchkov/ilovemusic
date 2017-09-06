# Music auto-tagging for humans

### Usage

#### extractor.py
-e <feature>          calculate feature, rewrite feature file
-u                    update base (add new songs)
-r                    rebuild base (recalculate all features)

#### classifier.py
-t <tag>  most relevant bands and songs for choosed tag
-a <artist> tag relevancy values for artist
-f <tag>  most important features for choosed tag
-s <tag1><sign><tag2> most relevant bands and songs for two tags combination; supported signs: "+", "-"
-p <artist> most important features for choosed artistle_playlist>.json  estimate class probability by playlist, JSON loaded from config.hand_class_path

### Avaliable tags
classical
dance
electronic
happy
melancholia
rock
sad
trash

### Basic extracted features
1. bpm
2. autocorrelation
3. zero-crossing
4. onset_strength
5. onset_regular
6. centroid
7. volume
8.1. spectral flux
8.2. spectral contrast
8.3. spectral std
8.4. spectral median
8.5. spectral bandwidth
8.6. spectral rolloff

### Details
Classification algorithm is gradient boosting. Feature selection algorithm is recursive feature elimination (RFE).
