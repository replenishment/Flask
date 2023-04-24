import pickle
from flask import Flask, render_template, request
import pandas as pd
from xgboost import XGBRegressor

app = Flask(__name__)

# Load the dataset
# df = pd.read_excel('C:/Users/chans/Downloads/VNG YY Data/Training Data - 111822 Reformat.xlsx')

# Get the unique values of the 'Brand' column
# unique_brands = df['SubGroup'].unique()

# Create a dictionary mapping each unique brand to an integer index
# brand_mapping = {brand: i for i, brand in enumerate(unique_brands)}

# Replace the Brand values with their Enum encoding
# df['SubGroup'] = df['SubGroup'].map(brand_mapping)

# y = df['ActualYY']
# X = df.drop(columns=["ID","ActualYY"])

# Define the hyperparameters to tune
params = {
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'tree_method':'hist',
    'enable_categorical': True,
}

# Train a Xgboost model
# xgb_model = XGBRegressor(**params)
# xgb_model.fit(X, y)

# Load the pickled model
with open('xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    # Get input values from form
    subgroup_predictor = int(request.form['subgroup'])
    pattern_predictor = int(request.form['pattern'])
    match_type_predictor = int(request.form['match_type'])
    ls_type_predictor = request.form.get('ls')
    selected_pcq = int(request.form['pcq'])
    selected_rx = int(request.form['rx'])
    selected_ry = int(request.form['ry'])
    selected_ans = int(request.form['ans'])
    selected_dc_type = request.form.get('dc')
    selected_mw = int(request.form['mw'])
    fit_type_predictor = int(request.form['fit_type'])
    
    
    if pattern_predictor == 1:
        solid_predictor = 1
        stripe_predictor = 0
        check_predictor = 0
    elif pattern_predictor == 2:
        solid_predictor = 0
        stripe_predictor = 1
        check_predictor = 0
    else:
        solid_predictor = 0
        stripe_predictor = 0
        check_predictor = 1
    
    if ls_type_predictor:
        ls_predictor = 1
        
    else:
        ls_predictor = 0
    
    
    if match_type_predictor == 1:
        owm_predictor = 1
        twm_predictor = 0    
    else: 
        owm_predictor = 0
        twm_predictor = 1
    
    if selected_dc_type:
        selected_dc = 1
        
    else:
        selected_dc = 0
    
  
    
    if fit_type_predictor == 1:
        cf_predictor = 1
        sf_predictor = 0
        esf_predictor = 0
    elif fit_type_predictor == 2:
        cf_predictor = 0
        sf_predictor = 1
        esf_predictor = 0
    else:
        cf_predictor = 0
        sf_predictor = 0
        esf_predictor = 1
    
        
    # Create an inputs dictionary
    inputs = {
        'SubGroup': [subgroup_predictor],
        'Pattern_Solid': [solid_predictor],
        'Pattern_Stripe': [stripe_predictor],
        'Pattern_Check': [check_predictor],
        'One_Way_Match': [owm_predictor],
        'Two_Way_Match': [twm_predictor],
        'Long_Sleeve': [ls_predictor],
        'Plan_Cut_Qty': [selected_pcq],
        'Repeat_X': [selected_rx],
        'Repeat_Y': [selected_ry],
        'Average_Neck_Size': [selected_ans],
        'Double_Cuff': [selected_dc],
        'Marker_Width': [selected_mw],
        'Regular_Fit': [cf_predictor],
        'Slim_Fit': [sf_predictor],
        'Extra_Slim_Fit': [esf_predictor],
    }

    # Convert inputs dictionary to DataFrame
    pred_df = pd.DataFrame(inputs)

    # Predict the yy
    y_output = xgb_model.predict(pred_df)
    output = round(y_output[0], 2)

    predicted_yy = request.args.get('output')
    return render_template('index.html', predicted_yy=y_output)
    
   
    

if __name__ == '__main__':
    app.run(debug=True)