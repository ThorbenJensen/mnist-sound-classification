import numpy as np
import librosa
import pandas as pd
import glob

SAMPLING_RATE = 8000

# %% INDEX RECORDINGS
data_folder = 'data/free-spoken-digit-dataset-1.0.8/recordings'

recordings = glob.glob(f'{data_folder}/*.wav')

# %% READ RECORDING
wavs = []
for rec in recordings:
    wavs.append(librosa.load(rec, sr=SAMPLING_RATE)[0])

# %% MAKE SAME LENGTH
lengths = [len(wav) for wav in wavs]
max_length = max(lengths)

wavs_padded = [np.pad(wav, (max_length,), 'constant', constant_values=0)
               for wav in wavs]

# %% FEATURE EXTRACTION
mfccs = [librosa.feature.mfcc(y=wav_padded, sr=SAMPLING_RATE, n_mfcc=40)
         for wav_padded in wavs_padded]

# %% DATA TO PANDAS
digits = [recording.split('/')[-1].split('_')[0] for recording in recordings]

df = pd.DataFrame({'digit': digits,
                   'mfcc': mfccs})

# %% WRITE RESULTS
df.to_parquet('data/df_preprocessed.parquet')
