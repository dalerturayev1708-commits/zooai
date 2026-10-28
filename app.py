from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model once when the server starts
model = joblib.load('zoo_model.joblib')

# Target names for the Iris dataset ()
target_names = ['Mammal', 'Bird', 'Reptile' , "Fish" , "Amphibian" , "Bug" , "Invertebrate"]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the 4 features from the form
        features = [float(x) for x in request.form.values()]
        
        if len(features) != 16:
            raise ValueError(
                f'Expected 16 features, got {len(features)}'
            )
        
        # The model expects a 2D array (1 sample, 4 features)
        final_features = np.array(features).reshape(1 , -1)
        
        # Make prediction
        prediction = model.predict(final_features)
        
        predicted_class = int(prediction[0])
        
        output = target_names[predicted_class]
        
        return render_template('index.html', prediction_text=f'Predicted zoo Species: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: Please enter valid numbers. ({e})')

if __name__ == "__main__":
    app.run(debug=True)
