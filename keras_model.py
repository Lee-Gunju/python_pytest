from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class SimpleKerasModel:
    def __init__(self):
        self.model = Sequential([
            Dense(64, activation='relu', input_shape=(1,)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X, y, epochs=100):
        self.model.fit(X, y, epochs=epochs, verbose=2)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = keras.models.load_model(file_path)