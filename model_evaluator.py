from sklearn.metrics import mean_squared_error

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X, y):
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        return mse