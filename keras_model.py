import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

class SimpleKerasModel:
    def __init__(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=100):
        self.model.fit(X, y, epochs=epochs, verbose=0)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)

if __name__ == "__main__":
    # 예제 데이터
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # 모델 생성 및 훈련
    model = SimpleKerasModel()
    model.train(X, y)

    # 모델 저장
    model.save_model('keras_model.h5')

    # 예측
    predictions = model.predict(np.array([[6], [7]]))
    print("Predictions:", predictions)