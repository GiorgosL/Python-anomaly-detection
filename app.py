import pandas as pd
import logging
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

import utils as ut

app = Flask(__name__)

ut.setup_logging()
config = ut.load_config(file_path='config.ini')
model = load_model(config['Settings']['model_name'])


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
           
            file = request.files['file']
            df = pd.read_csv(file)
            features = df.drop('Class', axis=1)
            target = df['Class']

            # Scale
            numeric_columns = features.select_dtypes(include='number').columns
            scaler = StandardScaler()
            features[numeric_columns] = scaler.fit_transform(features[numeric_columns])
            model_probs = model.predict(features)

            predictions = [(prob > 0.5).any() for prob in model_probs]
            df_probs = pd.DataFrame({'Prediction': predictions, 'Verdict': ['Fraud' if pred == 1 else 'Not Fraud' for pred in predictions]})

            return render_template('result.html', result=df_probs.to_html(classes='table table-striped'))

        except Exception as e:
            error_message = f"The following error occurred: {str(e)}"
            ut.log_message(error_message, level=logging.ERROR)
            return jsonify({"error": error_message})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
