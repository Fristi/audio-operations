import wave
import pickle
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
import librosa
from auditok import DataValidator, ADSFactory, DataSource, StreamTokenizer, BufferAudioSource

pylab.rcParams['figure.figsize'] = 24, 18

"""
Size of audio window for which MFCC coefficients are calculated
"""
ANALYSIS_WINDOW = 0.02 # 0.02 second = 20 ms

"""
Step of ANALYSIS_WINDOW 
"""
ANALYSIS_STEP = 0.01 # 0.01 second overlap between consecutive windows

"""
number of vectors around the current vector to return.
This will cause VectorDataSource.read() method to return
a sequence of (SCOPE_LENGTH * 2 + 1) vectors (if enough
data is available), with the current vetor in the middle
"""
SCOPE_LENGTH = 25

"""
Number of Mel filters
"""
MEL_FILTERS = 40

"""
Number of MFCC coefficients to keep
"""
N_MFCC = 16

"""
Sampling rate of audio data
"""
SAMPLING_RATE = 44100

"""
ANALYSIS_WINDOW and ANALYSIS_STEP as number of samples
"""
BLOCK_SIZE = int(SAMPLING_RATE * ANALYSIS_WINDOW)
HOP_SIZE = int(SAMPLING_RATE * ANALYSIS_STEP)

"""
Compute delta and delta-delta of MFCC coefficients ?
"""
DELTA_1 = True
DELTA_2 = True

"""
Where to find data
"""
PREFIX = "data/train"

def extract_mfcc(signal, sr=44100, n_mfcc=16, n_fft=256, hop_length=128, n_mels=40, delta_1=False, delta_2=False):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)

    if not (delta_1 or delta_2):
        return mfcc.T

    feat = [mfcc]

    if delta_1:
        mfcc_delta_1 = librosa.feature.delta(mfcc, order=1)
        feat.append(mfcc_delta_1)

    if delta_2:
        mfcc_delta_2 = librosa.feature.delta(mfcc, order=2)
        feat.append(mfcc_delta_2)

    return np.vstack(feat).T


def plot_signal_and_segmentation(signal, sampling_rate, segments=[]):
    _time = np.arange(0., np.ceil(float(len(signal))) / sampling_rate, 1. / sampling_rate)
    if len(_time) > len(signal):
        _time = _time[: len(signal) - len(_time)]

    pylab.subplot(211)

    for seg in segments:

        fc = seg.get("fc", "g")
        ec = seg.get("ec", "b")
        lw = seg.get("lw", 2)
        alpha = seg.get("alpha", 0.4)

        ts = seg["timestamps"]

        print("seg: {}, ts: {}", seg.get("title", ""), ts)

        # plot first segmentation outside loop to show one single legend for this class
        p = pylab.axvspan(ts[0][0], ts[0][1], fc=fc, ec=ec, lw=lw, alpha=alpha, label=seg.get("title", ""))

        for start, end in ts[1:]:
            p = pylab.axvspan(start, end, fc=fc, ec=ec, lw=lw, alpha=alpha)

    pylab.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                 borderaxespad=0., fontsize=22, ncol=2)

    pylab.plot(_time, signal)

    pylab.xlabel("Time (s)", fontsize=22)
    pylab.ylabel("Signal Amplitude", fontsize=22)
    pylab.show()


def file_to_mfcc(filename, sr=44100, **kwargs):
    print(filename)
    signal, sr = librosa.load(filename, sr=sr)

    return extract_mfcc(signal, sr, **kwargs)


class GMMClassifier():

    def __init__(self, models):
        """
        models is a dictionary: {"class_of_sound" : GMM_model_for_that_class, ...}
        """
        self.models = models

    def predict(self, data):
        result = []
        for cls in self.models:
            llk = self.models[cls].score_samples(data)[0]
            llk = np.sum(llk)
            result.append((cls, llk))

        """
        return classification result as a sorted list of tuples ("class_of_sound", log_likelihood)
        best class is the first element in the list
        """
        return sorted(result, key=lambda f: - f[1])


class ClassifierValidator(DataValidator):

    def __init__(self, classifier, target):
        """
        classifier: a GMMClassifier object
        target: string
        """
        self.classifier = classifier
        self.target = target

    def is_valid(self, data):
        r = self.classifier.predict(data)
        return r[0][0] == self.target


class VectorDataSource(DataSource):

    def __init__(self, data, scope=0):
        self.scope = scope
        self._data = data
        self._current = 0

    def read(self):
        if self._current >= len(self._data):
            return None

        start = self._current - self.scope
        if start < 0:
            start = 0

        end = self._current + self.scope + 1

        self._current += 1
        return self._data[start: end]

    def set_scope(self, scope):
        self.scope = scope

    def rewind(self):
        self._current = 0



train_data = {}

# train_data["speech"] = ["1.wav", "2.wav", "3.wav"]
train_data["fragments"] = ["1.wav", "2.wav", "3.wav", "4.wav"]
# train_data["silence"] = ["1.wav", "2.wav"]
# train_data["guitar"] = ["1.wav", "2.wav", "3.wav"]

models = {}

# build models
for cls in train_data:

    data = []
    for fname in train_data[cls]:
        mfcc = file_to_mfcc(PREFIX + '/' + cls + '/' + fname, sr=44100, n_mfcc=N_MFCC, n_fft=BLOCK_SIZE,
                            hop_length=HOP_SIZE, n_mels=MEL_FILTERS, delta_1=DELTA_1, delta_2=DELTA_2)

        data.append(mfcc)

    data = np.vstack(data)

    print("Class '{0}': {1} training vectors".format(cls, data.shape[0]))

    mod = GaussianMixture(n_components=10)
    mod.fit(data)
    # plt.plot(mod.means_)
    # plt.show()
    models[cls] = mod

# for cls in train_data:
#     fp = open("models/%s.gmm" % (cls), "wb")
#     pickle.dump(models[cls], fp, pickle.HIGHEST_PROTOCOL)
#     fp.close()

# models = {}
#
# for cls in ["speech", "guitar"]:
#     fp = open("models/%s.gmm" % (cls), "r")
#     models[cls] = pickle.load(fp)
#     fp.close()

gmm_classifier = GMMClassifier(models)

# create a validator for each sound class
# speech_validator = ClassifierValidator(gmm_classifier, "speech")
silence_validator = ClassifierValidator(gmm_classifier, "silence")
# guitar_validator = ClassifierValidator(gmm_classifier, "guitar")
fragment_validator = ClassifierValidator(gmm_classifier, "fragments")

file = "11_Minor_Pentatonic_Riffs.wav"
wfp = wave.open(file)
audio_data = wfp.readframes(-1)
width = wfp.getsampwidth()
wfp.close()
fmt = {1: np.int8 , 2: np.int16, 4: np.int32}
signal = np.array(np.frombuffer(audio_data, dtype=fmt[width]), dtype=np.float64)

mfcc_data_source = VectorDataSource(data=file_to_mfcc(file,
                                                      sr=44100, n_mfcc=N_MFCC,
                                                      n_fft=BLOCK_SIZE, hop_length=HOP_SIZE,
                                                      n_mels=MEL_FILTERS, delta_1=DELTA_1,
                                                      delta_2=DELTA_2), scope=SCOPE_LENGTH)

analysis_window_per_second = 1. / ANALYSIS_STEP

min_seg_length = 2 # second, min length of an accepted audio segment
max_seg_length = 20 # seconds, max length of an accepted audio segment
max_silence = 10 # second, max length tolerated of tolerated continuous signal that's not from the same class

segments = []

#fragment
tokenizer = StreamTokenizer(validator=fragment_validator, min_length=int(min_seg_length * analysis_window_per_second),
                                max_length=int(max_seg_length * analysis_window_per_second),
                                max_continuous_silence= max_silence * analysis_window_per_second,
                                mode = StreamTokenizer.DROP_TRAILING_SILENCE)

mfcc_data_source.rewind()
tokens = tokenizer.tokenize(mfcc_data_source)
speech_ts = [(t[1] * ANALYSIS_STEP, t[2] * ANALYSIS_STEP) for t in tokens]
seg = {"fc" : "r", "ec" : "r", "lw" : 0, "alpha" : 0.4, "title" : "Fragment (auto)", "timestamps" : speech_ts}
segments.append(seg)

# #silence
# tokenizer = StreamTokenizer(validator=silence_validator, min_length=int(min_seg_length * analysis_window_per_second),
#                                 max_length=int(max_seg_length * analysis_window_per_second),
#                                 max_continuous_silence= max_silence * analysis_window_per_second,
#                                 mode = StreamTokenizer.DROP_TRAILING_SILENCE)
#
# mfcc_data_source.rewind()
# tokenizer.validator = speech_validator
# tokens = tokenizer.tokenize(mfcc_data_source)
# silence_ts = [(t[1] * ANALYSIS_STEP, t[2] * ANALYSIS_STEP) for t in tokens]
# seg = {"fc" : "r", "ec" : "m", "lw" : 0, "alpha" : 0.4, "title" : "Silence (auto)", "timestamps" : silence_ts}
# segments.append(seg)



# plot automatic segmentation
plot_signal_and_segmentation(signal, SAMPLING_RATE, segments)