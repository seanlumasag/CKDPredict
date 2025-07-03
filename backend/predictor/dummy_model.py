class DummyCKDModel:
    def predict(self, X):
        results = []
        for features in X:
            age = features[0] 
            results.append(1 if age > 50 else 0)
        return results

model = DummyCKDModel()
