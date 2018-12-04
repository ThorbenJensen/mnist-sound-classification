from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras as k
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, \
    Flatten, Dense, BatchNormalization
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.models import Sequential

# %% LOAD DATA
df = pd.read_pickle('data/df_preprocessed.pkl')

dim1 = df.mfcc[0].shape[0]
dim2 = df.mfcc[0].shape[1]

INPUT_SHAPE = (dim1, dim2, 1)
NUM_CLASSES = df.digit.unique().size

# %% BUILD MODEL
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(48, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(120, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(loss=categorical_crossentropy,
              optimizer=k.optimizers.Adam(),
              metrics=['accuracy'])

# %% PRINT MODEL
outputs = [layer.output for layer in model.layers]
pprint(outputs)

# %% PARTITION TRAINING AND TESTING DATA

X = np.expand_dims(np.stack(df.mfcc.values, axis=0), -1)
y = k.utils.to_categorical(df.digit, NUM_CLASSES)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.1,
                                                    random_state=1)

# %% TRAIN MODEL
keras_callback = k.callbacks.TensorBoard(log_dir='./data/tf_log',
                                         histogram_freq=1,
                                         write_graph=True,
                                         write_images=True)

model.fit(X_train, y_train, batch_size=64, epochs=200, verbose=2,
          validation_split=0.1, callbacks=[keras_callback])

# %% EVALUATE ON TEST SET
score = model.evaluate(X_test, y_test, verbose=0)
model.predict_classes(X_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
