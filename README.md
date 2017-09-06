# ilovemusic

Automatic music classification for humans

### Usage

#### extractor.py
`-e <feature>` calculate feature, rewrite feature file

`-u` update base (add new songs)

`-r` rebuild base (recalculate all features)

#### classifier.py
`-t <tag>` most relevant bands and songs for choosed tag

`-a <artist>` tag relevancy values for artist

`-f <tag>` most important features for choosed tag

`-s <tag1><sign><tag2>` most relevant bands and songs
 for two tags combination; supported signs: "+", "-"

`-p <artist>` most important features for choosed artist

### Avaliable tags
classical, dance, electronic, happy, melancholia, rock, sad, trash

### Basic extracted features
1. bpm
2. autocorrelation
3. zero-crossing
4. onset_strength
5. onset_regular
6. centroid
7. volume
8. spectral flux
9. spectral contrast
10. spectral std
11. spectral median
12. spectral bandwidth
13. spectral rolloff
14. cepstral coefficients (n=10)

### Details
Classification algorithm is gradient boosting. Feature selection algorithm is recursive feature elimination (RFE).
