import glob #파일의 경로명을 이용해서 파일들의 리스트를 뽑음/인자로 받은 패턴과 이름이 일치하는 모든 파일과 디렉터리의 리스트를 반환
import os #운영체제와의 상호작용을 돕는 다양한 기능을 제공
import librosa #음원 데이터를 분석
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from keras.optimizers import Adam
#가중치규제추가도 생각해보자 4장42페이지
#validation도 필요
#가중치 초기화
#k폴드,앙상블

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name) #음원파일 불러옴-오디오시계열,샘플링속도
    stft = np.abs(librosa.stft(X))#푸리에 변환 후 절댓값
    #.T전치                               #오디오시계열,샘플링속도,반환할MFCC수
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)#노이즈제거?,특징추출
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)#파형 또는 전력 스펙트로 그램에서 크로마 그램을 계산
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0) #멜스케일된 스펙트럼을 계산
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0) #스펙트럼 비교?계산
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
                             #[]           []
    features, labels = np.empty((0,193)), np.empty(0) #값을 초기화 하지 않고 새로운 array만듬
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)): #파일 리스트를 가져옴(검색시 사용했던 경로명까지 전부 가져옴)
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn) #extract_feature함수불러옴
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz]) #배열을 가로로 결합(행 개수일치해야함)
            features = np.vstack([features,ext_features]) #배열을 세로로 결합(열 개수가 일치해야함)
            labels = np.append(labels, fn.split('\\')[2].split('-')[2])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels+1))
    one_hot_encode[np.arange(n_labels), labels] = 1
    one_hot_encode=np.delete(one_hot_encode, 0, axis=1)
    return one_hot_encode

#데이터 추출. 이미 데이터 추출되었으면 시간 오래 걸려서 생략
#change the main_dir acordingly....
# main_dir = 'C:/Audio_speech'
# sub_dir=os.listdir(main_dir)
# print ("\ncollecting features and labels...")
# print("\nthis will take some time...")
# features, labels = parse_audio_files(main_dir,sub_dir)
# print("done")
# np.save('X',features) #저장
# #one hot encoding labels
# labels = one_hot_encode(labels)
# np.save('y', labels)

X=np.load('X.npy')
y=np.load('y.npy')
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=60)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=60)

#층마다 차원
n_dim = train_x.shape[1] #193
n_classes = train_y.shape[1] #8
n_hidden_units_1 = n_dim
n_hidden_units_2 = 400

#모델 층
def create_model(activation_function='relu', init_type='normal', dropout_rate=0.25):
    model = Sequential()
    model.add(Dense(n_hidden_units_1, input_dim=n_dim, init=init_type, activation=activation_function))# 1
    model.add(Dense(n_hidden_units_2, init=init_type, activation=activation_function))  # 2
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes, init=init_type, activation='softmax')) # output
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.0005), metrics=['accuracy'])
    return model

model = create_model() #모델생성
epoch=150
train_history = model.fit(train_x, train_y, epochs=epoch, batch_size=10, validation_data=(val_x, val_y))
predict=model.predict(test_x,batch_size=4)
(test_loss, test_acc) = model.evaluate(test_x,  test_y, verbose=2)
print('\n테스트 정확도:', test_acc)

#테스트데이터로 감정예측
emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
y_pred = np.argmax(predict, 1)
predicted_emo=[]
for i in range(0,test_y.shape[0]):
  emo=emotions[y_pred[i]]
  predicted_emo.append(emo)

actual_emo=[]
y_true=np.argmax(test_y, 1)
for i in range(0,test_y.shape[0]):
  emo=emotions[y_true[i]]
  actual_emo.append(emo)

#그래프
epochs = range(1,epoch+1)
accuracy = train_history.history['accuracy']
val_accuracy = train_history.history['val_accuracy']
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
# t_acc = predict.history['accuracy']
# t_loss = predict.history['loss']

plt.subplot(1,2,1)
plt.plot(epochs, accuracy,'b', label='accuracy')
plt.plot(epochs, val_accuracy,'g', label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'r', label='loss')
plt.plot(epochs, val_loss, 'k', label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()

##그림안그려짐 왜??
# plt.subplot(2,2,3)
# plt.plot(epochs, t_acc, 'k', label='t_acc')
# plt.xlabel('Epochs')
# plt.ylabel('acc')
# plt.legend()
#
# plt.subplot(2,2,4)
# plt.plot(epochs, t_loss, 'k', label='t_loss')
# plt.xlabel('Epochs')
# plt.ylabel('loss')
# plt.legend()

#plt.grid(True)
plt.tight_layout()#떨어져있게 간격조정
plt.show()
