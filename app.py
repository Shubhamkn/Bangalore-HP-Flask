from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import math
from house_price_prediction_bangalore import *  # Import your existing Python code

app = Flask(__name__)

Bangalore_dataset.drop(['Price'], axis=1, inplace=True)
input_variables = list(Bangalore_dataset.columns)

location_option_list = list(le.classes_)

# print(input_variables)

# @app.route('/', methods=['POST','GET'])
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', input_variables = input_variables, location_option_list = location_option_list)

input_form = []

@app.route('/result/', methods=['POST'])
def result():
    # Collect user input from the form
    user_input = pd.DataFrame([request.form[variable] for variable in input_variables],index=input_variables)
    input_form = user_input.copy()
    # user_input.columns = input_variables.columns
    for var in input_variables:
        if (var=='Location'):
            val1 = le.transform([user_input.loc[var,0]])
            user_input.loc[var,0] = val1
        elif ((user_input.loc[var,0] == 'Yes') | (user_input.loc[var,0] =='No')):
            if (user_input.loc[var,0] == 'Yes'):
                user_input.loc[var,0] = 1
            else:
                user_input.loc[var,0] = 0

        user_input_final = user_input.reset_index()[0]
        
    np_arr = np.array(user_input_final)
    np_arr = np_arr.reshape(1,np_arr.shape[0])
    print(np_arr.shape)
    # Call the prediction function
    currVal = "Lakhs"
    p_price = models[3].predict(np_arr) / 100000 # converting the price in lakhs
    if p_price > 100:
        p_price = p_price / 100
        currVal = "Crores"
    rounded_price = np.round(p_price[0], 2)

    return render_template('result.html', predicted_price=rounded_price, input_form = input_form, Price_unit = currVal)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
