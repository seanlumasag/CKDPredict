class DummyCKDModel:
    def predict(self, X):
        # X is a list of input feature lists; return 0 or 1 for each
        # For example, just predict 1 (CKD risk) if age > 50, else 0
        results = []
        for features in X:
            age = features[0]  # assuming age is first feature
            results.append(1 if age > 50 else 0)
        return results

model = DummyCKDModel()
