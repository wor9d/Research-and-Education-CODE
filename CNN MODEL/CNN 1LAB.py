import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Seed 설정
np.random.seed(7)
tf.random.set_seed(7)

# 데이터 로드
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리
input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

static = []
static2 =[]
func1 =[]
func2 =[]
# 모델 생성
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
print(model.summary())

# 모델 컴파일 및 훈련
for i in range(12) :
  static2.append(i)
  static = [0] * 256
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  hist = model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(x_test, y_test))

  # 모델 평가
  score = model.evaluate(x_test, y_test, verbose=0)
  func1.append([i, score[0]])
  func2.append([i, score[1]])
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

  # 예측 및 시각화
  for n in range(256):
      predicted_class = np.argmax(model.predict(x_test[n].reshape((1, 28, 28, 1))), axis=-1)
      true_class = np.argmax(y_test[n])  # 원-핫 인코딩을 정수로 변환

      if predicted_class[0] == true_class:
          static[n] = 1
      else:
        static2.append([n,np.argmax(y_test[n]), predicted_class[0]])


        plt.imshow(x_test[n].reshape(28, 28), cmap='gray')  # 이미지가 흑백일 경우 cmap='gray'
        plt.axis('off')
        plt.show()

  print(sum(static))
  print(static2)

def function(t , f):
  points = np.array(t)
  x = points[:, 0]
  y = points[:, 1]

  # 곡선 보간
  x_new = np.linspace(x.min(), x.max(), 300)  # 곡선의 매끄러움을 위한 더 많은 x 좌표 생성
  spline = interp1d(x, y, kind='cubic')       # cubic은 부드러운 곡선을 위한 옵션
  y_smooth = spline(x_new)

  # 그래프 그리기
  plt.plot(x, y, 'o', label='Data Points')  # 원래 데이터 포인트
  plt.plot(x_new, y_smooth, '-', label='Interpolated Curve')  # 보간된 곡선
  plt.legend()
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title(f)
  plt.show()

function(func1, 'loss')
function(func2, 'accuracy')

def extract(alist):
    elements = set()

    for item in alist:
        if isinstance(item, list) and item:  # 배열이면서 비어 있지 않을 경우
            elements.add(item[0])
    return list(elements)

image = extract(static2)
for n in image:
  plt.imshow(x_test[n].reshape(28, 28), cmap='gray')  # 이미지가 흑백일 경우 cmap='gray'
  plt.axis('off')
  plt.show()
