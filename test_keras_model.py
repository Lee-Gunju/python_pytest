import numpy as np
from keras_model import SimpleKerasModel

def test_model_training_and_prediction():
    # 예제 데이터
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # 모델 생성 및 훈련
    model = SimpleKerasModel()
    model.train(X, y, epochs=100)  # epochs 수를 늘림

    # 예측
    predictions = model.predict(np.array([[6], [7]]))

    # 예측 값이 기대 범위 내에 있는지 확인
    assert np.allclose(predictions, np.array([[12], [14]]), atol=1.0)