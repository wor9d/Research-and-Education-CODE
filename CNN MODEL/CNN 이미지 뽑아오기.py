import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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

# 예측 및 시각화
from PIL import Image

# 0번부터 255번까지의 테스트 이미지를 선택하고 64x64로 리사이즈
selected_images = x_test[:256]
resized_images = [Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)).resize((32, 32), Image.LANCZOS) for img in selected_images]

# 빈 512x512 캔버스 생성
canvas_size = (512, 512)
canvas = Image.new('L', canvas_size)

# 8x8 그리드에 이미지를 배치
for i in range(16):
    for j in range(16):
        img = resized_images[i * 8 + j]
        canvas.paste(img, (j * 32, i * 32))

# 결과 이미지 보여주기
plt.imshow(canvas, cmap='gray')
plt.axis('off')
plt.show()

# 결과 이미지 저장
canvas.save("/content/mnist_test_512x512.png") #보고서용 이미지입니다.
