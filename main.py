import numpy as np
from data_preprocessor import DataPreprocessor
from keras_model import SimpleKerasModel
from model_evaluator import ModelEvaluator

if __name__ == "__main__":
    # 예제 데이터
    data = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])

    # 데이터 전처리
    preprocessor = DataPreprocessor(data)
    normalized_data = preprocessor.normalize()
    X, y = preprocessor.get_features_and_labels()

    # 모델 생성 및 훈련
    model = SimpleKerasModel()
    model.train(X, y)

    # 모델 평가
    evaluator = ModelEvaluator(model)
    mse = evaluator.evaluate(X, y)
    print("Mean Squared Error:", mse)

    # 모델 저장
    model.save_model('keras_model.h5')

    # 사용자 입력을 받아 예측
    while True:
        user_input = input("Enter a number to predict (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        try:
            value = float(user_input)
            prediction = model.predict(np.array([[value]]))
            print(f"Prediction for {value}: {prediction[0][0]}")
        except ValueError:
            print("Please enter a valid number.")