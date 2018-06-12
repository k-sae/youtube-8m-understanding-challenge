from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import LSTM, Dense

from reader import read_data


def get_model():
    my_model = Sequential()
    my_model.add(Dense(1024, activation="relu", input_shape=(1152,)))
    my_model.add(Dense(2048, activation="relu"))
    my_model.add(Dense(4096, activation="relu"))
    my_model.add(Dense(3862, activation="sigmoid"))
    my_model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])
    return  my_model


train = read_data("datasets/video_sample/train00.tfrecord")
validation = read_data("datasets/video_sample/train01.tfrecord")

model = get_model()
early_stop = EarlyStopping(patience=4, monitor='val_loss')
checkpoint = ModelCheckpoint("weights.h5",
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             )
csv_logger = CSVLogger('v.csv')
model.fit_generator(train,
                    steps_per_epoch=50,
                    epochs=50,
                    validation_data=validation,
                    validation_steps=20, callbacks=[early_stop, checkpoint, csv_logger])

