from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():

    # input given model
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')
    # start testing code

# here we are storing above data(obtaind from form) into dictionay foram
    # result = {'cgpa': cgpa, 'iq': iq, 'profile_score': profile_score}

# result are retured back in json format(universal language)
# for doing all this bhasar postman is helping us[postman is helpful in testing of url]
    # return jsonify(result)

# End testing code
    input_query = np.array([[cgpa, iq, profile_score]])

    result = model.predict(input_query)[0]

    return jsonify({'placement': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
