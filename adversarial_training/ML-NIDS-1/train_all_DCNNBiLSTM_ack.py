import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress INFO and WARNING logs but keep errors
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf

# Drastically reduce thread usage
tf.config.threading.set_intra_op_parallelism_threads(1)  # Computation threads
tf.config.threading.set_inter_op_parallelism_threads(1)  # Parallelism across ops

from tensorflow import keras
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

from matplotlib import pyplot as plt
import seaborn as sns

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    print(gpu, "\n")
else:
  print("No GPU device found")

################
print("TensorFlow is using GPU:", tf.test.is_built_with_cuda())
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Check the current device being used
print("Device in use:")
print(tf.config.experimental.list_logical_devices('GPU'))

################

# Load dataset
df =pd.read_csv('./Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv', low_memory=False)
print(df.shape)

df = df[~((df['tcp.flags'] == 16) & (df['tcp.len'] == 0))]
print(df.shape)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Read Completed!")
print(df.head(5))
print(df['Attack_type'].value_counts())

from sklearn.utils import shuffle

drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4",
                "http.file_data","http.request.full_uri","icmp.transmit_timestamp",
                "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",
                "tcp.dstport", "udp.port", "mqtt.msg"]

print(len(df.columns))
df.drop(drop_columns, axis=1, inplace=True, errors='ignore')
df.dropna(axis=0, how='any', inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)
df = shuffle(df)
df.isna().sum()
print(len(df.columns))
print(df['Attack_type'].value_counts())

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

categorical_features = ['http.request.method', 'http.referer', 'http.request.version',
                        'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic']

for feature in categorical_features:
    encode_text_dummy(df, feature)

#df.to_csv('./Selected dataset for ML and DL/DNN-EdgeIIoT-dataset_processed.csv', encoding='utf-8', index=False)

#df = pd.read_csv('./Selected dataset for ML and DL/DNN-EdgeIIoT-dataset_processed.csv', low_memory=False)

df['Attack_type'].value_counts()

feat_cols = list(df.columns)
label_col = "Attack_type"

feat_cols.remove(label_col)
print(feat_cols)
print(len(feat_cols))

empty_cols = [col for col in df.columns if df[col].isnull().all()]
print(empty_cols)

corr_matrix = df[feat_cols].corr()
fig = plt.figure(figsize=(50,50))
sns.heatmap(corr_matrix, annot=True)

skip_list = ["icmp.unused", "http.tls_port", "dns.qry.type", "mqtt.msg_decoded_as"]
print(df[skip_list[3]].value_counts())

df.drop(skip_list, axis=1, inplace=True)
feat_cols = list(df.columns)

import joblib

joblib.dump(feat_cols, 'feature_columns_ack.pkl')
feat_cols.remove(label_col)
print(feat_cols)

X = df.drop([label_col], axis=1)
y = df[label_col]
del df

import pickle
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

del X
del y

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
Y_train =  label_encoder.fit_transform(y_train)
Y_test = label_encoder.transform(y_test)

# Save the LabelEncoder
with open('label_encoder_ack.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print(label_encoder.classes_)

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=np.unique(y_train),
                                                 y=y_train)

class_weights = {k: v for k,v in enumerate(class_weights)}
print(class_weights)

#from imblearn.over_sampling import SMOTE

#oversample = SMOTE()
#X_train, Y_train = oversample.fit_resample(X_train, Y_train)

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
X_train =  min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

# Save the MinMaxScaler
with open('min_max_scaler_ack.pkl', 'wb') as f:
    pickle.dump(min_max_scaler, f)

input_shape = X_train.shape[1]

print(X_train.shape, X_test.shape)
print(input_shape)

num_classes = len(np.unique(Y_train))
print(num_classes)

from  tensorflow.keras.utils import to_categorical

Y_train = to_categorical(Y_train, num_classes=num_classes)
Y_test = to_categorical(Y_test, num_classes=num_classes)

print(Y_train.shape, Y_test.shape)

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Input, ZeroPadding1D
from tensorflow.keras.layers import MaxPooling1D, Add, AveragePooling1D
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam

from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import LSTM, Bidirectional, BatchNormalization, Convolution1D, MaxPooling1D, Reshape

#Bidirectional CNN-LSTM
batch_size = 32
model = Sequential()

model.add(Convolution1D(96, kernel_size=76, padding="same",activation="relu",input_shape=(92, 1)))
#model.add(MaxPooling1D(pool_length=(5)))

model.add(BatchNormalization())

model.add(Bidirectional(LSTM(96, return_sequences=False)))
model.add(Reshape((192, 1), input_shape = (96, )))

model.add(BatchNormalization())

model.add(Bidirectional(LSTM(128, return_sequences=False)))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(15)) #Output Layer
model.add(Activation('softmax'))

model.summary()

#optimizer
opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print("Starting to train")
early_stop_callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.0001)

history = model.fit(X_train, Y_train,
                              batch_size=128,
                              epochs=30,
                              verbose=True, #callbacks=[reduce_lr, early_stop_callback],
                              validation_data=(X_test, Y_test))

model_name = f'my_model_all_in_one_ack.h5'
model.save(model_name)
print(f"Model saved as {model_name}")

