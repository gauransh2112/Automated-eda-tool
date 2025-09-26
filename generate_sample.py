# generate_sample.py
from sklearn.datasets import load_iris
import pandas as pd
import os

os.makedirs("examples", exist_ok=True)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=[c.replace(" (cm)", "").replace(" ", "_") for c in iris.feature_names])
df["target"] = iris.target
df.to_csv("examples/iris.csv", index=False)
print("Wrote examples/iris.csv")
