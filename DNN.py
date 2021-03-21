####    Importing Libraries    ####

from __future__ import print_function
import pandas as pd
import numpy as np
import sys
import sklearn
import io
import random
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

####    Specify Datasets    ####

train_url = 'https://raw.githubusercontent.com/og124/NSL-KDD-Deep-Neural-Network/main/NSL-KDD-Dataset/KDDTrain.csv'
test_url = 'https://raw.githubusercontent.com/og124/NSL-KDD-Deep-Neural-Network/main/NSL-KDD-Dataset/KDDTest.csv'

####    Define Column Names of Datasets     ####

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

####    Assign Columns     ####

df = pd.read_csv(train_url,header=None, names = col_names)
df_test = pd.read_csv(test_url, header=None, names = col_names)

####    Catergories Into Array  ####

categorical_columns=['protocol_type', 'service', 'flag']

df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]

####    Dummy Columns   ####

unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]

dumcols=unique_protocol2 + unique_service2 + unique_flag2

####    Categories Array into Numbers via LabelEncoder  ####

unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2

df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)

####    OHE     ####

enc = OneHotEncoder(categories='auto')
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

#### Define Encoded Data into Variable for Modelling    ####

traindata = df_cat_data
testdata = testdf_cat_data

X = traindata.iloc[:,0:42]
Y = traindata.iloc[:,42]
C = testdata.iloc[:,42]
T = testdata.iloc[:,0:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.array(trainX)
X_test = np.array(testT)

batch_size = 64

####    Define Neural Network   ####

model = Sequential()
model.add(Dense(1024,input_dim=(4),activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(768,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(512,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.01))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
checkpointer = callbacks.ModelCheckpoint(filepath="nslresults/dnn4layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('nslresults/dnn4layer/training_set_dnnanalysis.csv',separator=',', append=False)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),batch_size=batch_size, epochs=100, callbacks=[checkpointer,csv_logger])
model.save("nslresults/dnn4layer/dnn4layer_model.hdf5")


