from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_path = Path(__file__).with_name("student_data.csv")
df = pd.read_csv(data_path, sep=";")
print(df)
print("Dataset shape:", df.shape)
print(df.describe())

plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="G3", bins=20, kde=True, color="blue")
plt.title("Final grade")
plt.xlabel("Final grade")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x="studytime", y="G3", data=df, palette="Set2")
plt.title("Grade across study time levels")
plt.xlabel("Study time")
plt.ylabel("Final grade")
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x="G1", y="G3", data=df, hue="sex", palette="coolwarm")
plt.title("First period grade vs Final grade")
plt.xlabel("G1")
plt.ylabel("G3")
plt.show()

sns.pairplot(
    df[["age", "studytime", "G1", "G2", "G3", "sex"]],
    hue="sex",
    palette="Set1",
)
plt.suptitle("Pairwise relationships", y=1.02)
plt.show()

corr_matrix = df[["age", "studytime", "G1", "G2", "G3"]].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation")
plt.show()