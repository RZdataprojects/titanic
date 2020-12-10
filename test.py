import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

plt.style.use('ggplot')

df = pd.read_csv(r"C:\Users\Roei\source\repos\titanic\DB\train.csv")
print(df.groupby("Survived").size())

x = df[['Fare', 'Embarked']]
y = df["Survived"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1, stratify=y)
# test_size=1 because x & y are assigned data from a train file - test file is seperated

# build a dict mapping species to an integer code
inv_name_dict = {0: 'Survived', 1: 'Perished'}

# Create ML Model
knn = KNeighborsClassifier(n_neighbors=5)
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, test_size=0.30, random_state=1, stratify=y)
# instantiate
knn2 = KNeighborsClassifier(n_neighbors=5)
# fit
knn2.fit(x_train1, y_train1)
