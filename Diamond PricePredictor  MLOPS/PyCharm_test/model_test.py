import pickle
import pandas as pd
import numpy as np

with open("../The_Model.pkl", "rb") as f:
    saved_data = pickle.load(f)

model = saved_data["model"]
X_test_scaled = pd.read_csv("../testdatascaled.csv")
print(model.predict(X_test_scaled))

