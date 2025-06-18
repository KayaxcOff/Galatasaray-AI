import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

dataBase = pd.read_excel("ai_project.xlsx")

x = dataBase.drop("Oyuncular", axis=1)
y = dataBase["Oyuncular"]

x = pd.get_dummies(x)
y = pd.get_dummies(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(x.columns), "x_columns.pkl")

model = Sequential()
model.add(Dense(8, activation="relu", input_dim=x.shape[1]))
model.add(Dense(4, activation="relu"))
model.add(Dense(y.shape[1], activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=10, batch_size=4, validation_split=0.2)

model.save("gs_ai_model.keras")
