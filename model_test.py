from tensorflow.keras.models import load_model
import pandas as pd
import joblib

model = load_model("gs_ai_model.keras")
scaler = joblib.load("scaler.pkl")

guessThePosition = str(input("Tell us the position the player plays: "))
guessTheCountry = str(input("Tell us the country the player come from: "))
guessTheAge = int(input("What is the player's age: "))
guessTheNumber = int(input("What is the player's number: "))
guessTheFoot = str(input("Which foot the player uses: "))
guessTheMatch = int(input("How much match the player played: "))
guessTheGoal = int(input("How much goal the player scored: "))
guessTheAsist = int(input("How much goal the player helped: "))

newDataBase = pd.DataFrame([{
    "Mevki" : guessThePosition,
    "Uyruk" : guessTheCountry,
    "Yaş" : guessTheAge,
    "Forma numarası" : guessTheNumber,
    "Kullandığı ayak" : guessTheFoot,
    "Maç sayısı" : guessTheMatch,
    "Gol" : guessTheGoal,
    "Asist" : guessTheAsist
}])

newDataBase = pd.get_dummies(newDataBase)

dataBaseForGuess = pd.read_excel("guessThePlayer.xlsx")
for col in dataBaseForGuess.columns:
    if col not in newDataBase.columns:
        newDataBase[col] = 0
newDataBase = newDataBase[dataBaseForGuess.columns]

newDataBase_scaled = scaler.transform(newDataBase)

prediction = model.predict(newDataBase_scaled)

import numpy as np
player_index = np.argmax(prediction)
print("Tahmin edilen oyuncu indexi:", player_index)
print("Olasılık dağılımı:", prediction)
