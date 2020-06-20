import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler

sns.set_style("whitegrid")
pd.set_option('display.max_rows', 800)
os.getcwd()
print('Import of the packages Successful')
np.random.seed(697)
# Water Treatment Dataset
water_treatment_dataset = pd.read_csv("water-treatment.data", sep=",", header=None)
# Adding to a Data frame
df = pd.DataFrame(water_treatment_dataset)
df.columns = ['Date', 'Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P',
              'SS-P', 'SSV-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S',
              'DBO-S', 'DQO-S', 'SS-S', 'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S',
              'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G']
# Replace ? with NaN values
data_frame = df.replace('?', np.nan)
#print(data_frame)
# Converting following Data frame features to numeric
data_frame[
    ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P', 'SS-P', 'SSV-P',
     'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SS-S',
     'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S', 'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G',
     'RD-SS-G', 'RD-SED-G']] = data_frame[
    ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P', 'SS-P', 'SSV-P',
     'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SS-S',
     'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S', 'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G',
     'RD-SS-G', 'RD-SED-G']].apply(pd.to_numeric)
# Testing Data types and NaN values
#print(data_frame.dtypes)
column_names = data_frame.columns

# Creating list of features in the Dataset
column_list = ['Date', 'Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P',
               'SS-P', 'SSV-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S',
               'DBO-S', 'DQO-S', 'SS-S', 'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S',
               'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G']

# Filling the NaN values with Median
i = 1
while i < len(column_list):
    median = data_frame[column_list[i]].median()
    data_frame[column_list[i]].fillna(median, inplace=True)
    i += 1
#print(data_frame)

# Dropping Date Column for the Dataset to be normalized
data_frame.drop(data_frame.columns[0], axis=1, inplace=True)
#print(data_frame)

scaler = MinMaxScaler()
data_frame_norm = scaler.fit_transform(data_frame)
data_frame_norm = pd.DataFrame(data_frame_norm)
data_frame_norm = data_frame_norm.values
train, test_df = train_test_split(data_frame_norm, test_size=0.15, random_state=42)
train_df, dev_df = train_test_split(train, test_size=0.15, random_state=42)
train_df.sum() / train_df.shape[0]
dev_df.sum() / dev_df.shape[0]
test_df.sum() / test_df.shape[0]

train_y = train_df
dev_y = dev_df
test_y = test_df

train_x = train_df
dev_x = dev_df
test_x = test_df

train_x = np.array(train_x)
dev_x = np.array(dev_x)
test_x = np.array(test_x)

train_y = np.array(train_y)
dev_y = np.array(dev_y)
test_y = np.array(test_y)

encoding_dim = 16

input_data = Input(shape=(train_x.shape[1],))

encoded = Dense(encoding_dim, activation='elu')(input_data)

decoded = Dense(train_x.shape[1], activation='sigmoid')(encoded)

autoencoder = Model(input_data, decoded)

autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy')

hist_auto = autoencoder.fit(train_x, train_x,
                            epochs=50,
                            batch_size=30,
                            shuffle=True,
                            validation_data=(dev_x, dev_x))

plt.figure()
plt.plot(hist_auto.history['loss'])
plt.plot(hist_auto.history['val_loss'])
plt.title('Autoencoder model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

encoder = Model(input_data, encoded)

encoded_input = Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input, decoder_layer(encoded_input))

encoded_x = encoder.predict(test_x)
decoded_output = decoder.predict(encoded_x)

encoded_train_x = encoder.predict(train_x)
encoded_test_x = encoder.predict(test_x)

model = Sequential()
model.add(Dense(16, input_dim=encoded_train_x.shape[1],
                kernel_initializer='normal',
                # kernel_regularizer=regularizers.l2(0.02),
                activation="relu"
                )
          )

model.add(Dropout(0.2))

model.add(Dense(38))

model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer='adam')

history = model.fit(encoded_train_x, train_y, validation_split=0.2, epochs=10, batch_size=64)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Encoded model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

predictions_NN_prob = model.predict(encoded_test_x)
predictions_NN_prob = predictions_NN_prob[:, 0]
predictions_NN_01 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_02 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_03 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_04 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_05 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_06 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_07 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_08 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_09 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_10 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_11 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_12 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_13 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_14 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_15 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_16 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_17 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_18 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_19 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_20 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_21 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_22 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_23 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_24 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_25 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_26 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_27 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_28 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_29 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_30 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_31 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_32 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_33 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_34 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_35 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_36 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_37 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_38 = np.where(predictions_NN_prob > 0.5, 1, 0)
predictions_NN_39 = np.where(predictions_NN_prob > 0.5, 1, 0)
