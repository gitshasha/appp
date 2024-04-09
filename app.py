from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

@app.route('/'  )
def home():
    return render_template("index.html")




@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.json
# Load the pickled model
    model=pickle.load(open('./loan_ada_model.pkl', 'rb'))
    print("thus",data['data'])
    
    # Preprocess input data if necessary
    # Example: Convert data to the format expected by your model

    # Make prediction using the model
    prediction = model.predict(data['data'])
   
    ans=prediction[0]
    # Return prediction as JSON response
    return jsonify({'prediction':str(ans) })

if __name__ == '__main__':
    app.run(debug=True)
