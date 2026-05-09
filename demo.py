import pandas as pd

df = pd.read_csv("KDDTrain+.txt", header=None)

demo_df = df.sample(1000, random_state=42)

demo_df.to_csv("demo_dataset.csv", index=False, header=False)

print("Demo dataset created")