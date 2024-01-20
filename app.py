import numpy as np
import logging
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model


import utils as ut

app = Flask(__name__)

ut.setup_logging()
config = ut.load_config(file_path='config.ini')
model = load_model(config['Settings']['model_name'])

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Assuming the input data is in JSON format
        data = request.get_json()
        csv_data = data.get('csvData')

        # Convert CSV data to a Pandas DataFrame
        df = pd.read_csv(pd.compat.StringIO(csv_data))

        # Perform any necessary preprocessing on features
        # ...

        # Make predictions using the loaded autoencoder model
        predictions = model.predict(df)

        # Return the predictions as JSON
        return jsonify(predictions.tolist())

    except Exception as e:
        ut.log_message(f"The following occured: {str(e)}", level=logging.ERROR)

if __name__ == '__main__':
    app.run(debug=True)
