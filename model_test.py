from tensorflow.keras.models import load_model
import pandas as pd
import joblib
import numpy as np

model = load_model("gs_ai_model.keras")
scaler = joblib.load("scaler.pkl")
x_columns = joblib.load("x_columns.pkl")

guessThePosition = str(input("Tell us the position the player plays: "))
guessTheCountry = str(input("Tell us the country the player come from: "))
guessTheAge = int(input("What is the player's age: "))
guessTheNumber = int(input("What is the player's number: "))
guessTheFoot = str(input("Which foot the player uses: "))
guessTheMatch = int(input("How many matches the player played: "))
guessTheGoal = int(input("How many goals the player scored: "))
guessTheAssist = int(input("How many assists the player has: "))

newDataBase = pd.DataFrame([{
    "Mevki" : guessThePosition,
    "Uyruk" : guessTheCountry,
    "Yaş" : guessTheAge,
    "Forma numarası" : guessTheNumber,
    "Kullandığı ayak" : guessTheFoot,
    "Maç sayısı" : guessTheMatch,
    "Gol" : guessTheGoal,
    "Asist" : guessTheAssist
}])

newDataBase = pd.get_dummies(newDataBase)

for col in x_columns:
    if col not in newDataBase.columns:
        newDataBase[col] = 0

newDataBase = newDataBase[x_columns]

newDataBase_scaled = scaler.transform(newDataBase)

prediction = model.predict(newDataBase_scaled)

player_index = np.argmax(prediction)
print("Index of the player:", player_index)
print("probability distribution:", prediction)
