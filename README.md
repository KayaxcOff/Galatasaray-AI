# Galatasaray-AI

Galatasaray-AI is a beginner-friendly artificial intelligence project that aims to predict which Galatasaray football player matches the given attributes. This project leverages machine learning techniques to make predictions based on past season data of Galatasaray players. It is built using Python, TensorFlow, pandas, and scikit-learn.

## Project Purpose

The goal of this project is to predict which player is being described, given user inputs such as age, position, jersey number, matches played, goals, assists, nationality, and preferred foot. The user provides these features and the model tries to match them to a player in the dataset.

## Files Included

- `ai_project.xlsx`: The main dataset containing player statistics and attributes.
- `create_model.py`: Code for training the machine learning model.
- `model_test.py`: Code for testing or making predictions with the trained model.
- `gs_ai_model.keras`: The trained Keras model file.
- `scaler.pkl`: The scaler object used for feature normalization.
- `x_columns.pkl`: List of feature columns after one-hot encoding.

## Installation & Usage

### Requirements

- Python 3.x
- TensorFlow
- pandas
- scikit-learn
- joblib

Install the required libraries with:

```bash
pip install tensorflow pandas scikit-learn joblib
```

### Running the Model

1. Make sure all required files (model, scaler, x_columns, dataset) are in the same directory.
2. Run `model_test.py` or use your own script to make predictions. Example flow:

### Output

The script prints the index of the player with the highest probability and the raw probability distribution.

## About the Dataset

> **Important Note:**  
> The dataset used in this project is very small and only includes a limited number of Galatasaray players from past seasons.  
> **Due to the small dataset size, the model's accuracy is low, and it may not predict the correct player even when provided with exact feature values from the dataset.**  
> For reliable AI models, a much larger and more diverse dataset is necessary.

## Known Issues

- The model tends to overfit and does not generalize well, due to the tiny dataset.
- Categorical feature values (e.g., nationality, position, footedness) must match exactly between training and user input (e.g., "Uruguay" vs "Urugay" will not match).
- Only works for a limited set of players/data.

## Contributions & License

This project is for educational and experimental purposes. Contributions are welcome via pull requests.

---

**Author:** [KayaxcOff](https://github.com/KayaxcOff)
