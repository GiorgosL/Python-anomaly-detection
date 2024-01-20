import logging
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

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
            df_probs = ut.get_predictions(file=file, model=model)
            print(df_probs) 
            return render_template('result.html', result=df_probs.to_html(classes='table table-striped'))
        
        except Exception as e:
            ut.log_message(f'The following occured: {str(e)}', level=logging.ERROR)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
