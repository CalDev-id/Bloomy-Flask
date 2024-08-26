from tensorflow.keras.models import load_model

# Load the original model
model = load_model("models/marine_price.h5")

sample_input = [[1,1,1,1]] 

# Use the trained model to predict
predicted_class = model.predict(sample_input)

print("Predicted class:", predicted_class)