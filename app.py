from flask import Flask, request, jsonify
from flask_cors import CORS 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import joblib

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# ------------------- Data Preparation and Model Training ------------------- #
df = pd.read_csv('./Salary_Data.csv')

# Function to classify job titles into Job_Level and Job_Field
def classify_job(title):
    # Ensure that title is a string before applying any operations
    if not isinstance(title, str):
        title = str(title)  # Convert to string if it's not a string
    
    job_levels = {'Senior': 'Senior', 'Junior': 'Junior', 'Director': 'Senior', 'Manager': 'Mid'}
    job_fields = {
        'Software': 'IT', 'Data': 'Data', 'Marketing': 'Marketing', 'HR': 'Human Resources',
        'Operations': 'Operations', 'Product': 'Product Design'
    }
    level = 'Other'
    for key, value in job_levels.items():
        if key in title:
            level = value
            break
    field = 'General'
    for key, value in job_fields.items():
        if key in title:
            field = value
            break
    return level, field

# Apply classification to the dataset
df[['Job_Level', 'Job_Field']] = df['Job Title'].apply(lambda x: pd.Series(classify_job(x)))

# ------------------- Handle Missing Data ------------------- #

# Replace NaN in Gender with 'Unknown' before encoding
df['Gender'] = df['Gender'].fillna('Unknown')

# Fill missing values in Education Level, Job_Level, and Job_Field
df['Education Level'] = df['Education Level'].fillna('Unknown')
df['Job_Level'] = df['Job_Level'].fillna('Other')
df['Job_Field'] = df['Job_Field'].fillna('General')

# Drop rows where target (Salary) is NaN
df = df.dropna(subset=['Salary'])

# ------------------- Encode Categorical Features ------------------- #

# Explicit categories for Gender
label_encoder_gender = LabelEncoder()
label_encoder_gender.fit(['Male', 'Female', 'Unknown'])

# Explicit categories for Education Level (including 'Bachelor's' and 'Master's')
label_encoder_edu = LabelEncoder()
label_encoder_edu.fit(["Bachelor's", "Master's", "PhD", 'Unknown'])  # Include Bachelor's and Master's

# Job level and job field encoding
label_encoder_job_level = LabelEncoder()
label_encoder_job_field = LabelEncoder()

df['Gender'] = label_encoder_gender.transform(df['Gender'])
df['Education Level'] = label_encoder_edu.transform(df['Education Level'])
df['Job_Level'] = label_encoder_job_level.fit_transform(df['Job_Level'])
df['Job_Field'] = label_encoder_job_field.fit_transform(df['Job_Field'])

# ------------------- Feature Selection and Target ------------------- #

X = df[['Age', 'Gender', 'Education Level', 'Years of Experience', 'Job_Level', 'Job_Field']]
y = df['Salary']

# Impute missing values in features
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model trained successfully!")
print(f"R-squared Score: {r2_score(y_test, y_pred)}")

# Save the model, label encoders, and imputer
joblib.dump(model, "salary_predictor_model.pkl")
joblib.dump(label_encoder_gender, "label_encoder_gender.pkl")
joblib.dump(label_encoder_edu, "label_encoder_edu.pkl")
joblib.dump(label_encoder_job_level, "label_encoder_job_level.pkl")
joblib.dump(label_encoder_job_field, "label_encoder_job_field.pkl")
joblib.dump(imputer, "imputer.pkl")

# ------------------- Flask API Endpoint ------------------- #

# Load the model and other components
model = joblib.load("salary_predictor_model.pkl")
label_encoder_gender = joblib.load("label_encoder_gender.pkl")
label_encoder_edu = joblib.load("label_encoder_edu.pkl")
label_encoder_job_level = joblib.load("label_encoder_job_level.pkl")
label_encoder_job_field = joblib.load("label_encoder_job_field.pkl")
imputer = joblib.load("imputer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Ensure all required keys exist in the input data
    required_keys = ['age', 'gender', 'education', 'experience', 'job_level', 'job_field']
    if not all(key in data for key in required_keys):
        return jsonify({'error': 'Missing required input fields'}), 400
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [data['age']],
        'Gender': [data['gender']],
        'Education Level': [data['education']],
        'Years of Experience': [data['experience']],
        'Job_Level': [data['job_level']],
        'Job_Field': [data['job_field']]
    })
    
    # Replace missing input values with 'Unknown' for categorical fields
    input_data['Gender'] = input_data['Gender'].fillna('Unknown')
    input_data['Education Level'] = input_data['Education Level'].fillna('Unknown')
    input_data['Job_Level'] = input_data['Job_Level'].fillna('Other')
    input_data['Job_Field'] = input_data['Job_Field'].fillna('General')
    
    # Transform input categorical values using label encoders
    try:
        input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
        input_data['Education Level'] = label_encoder_edu.transform(input_data['Education Level'])
        input_data['Job_Level'] = label_encoder_job_level.transform(input_data['Job_Level'])
        input_data['Job_Field'] = label_encoder_job_field.transform(input_data['Job_Field'])
    except ValueError as e:
        return jsonify({'error': f'Invalid input value: {str(e)}'}), 400

    # Impute missing numerical values
    input_data_imputed = imputer.transform(input_data)

    # Predict the salary
    prediction = model.predict(input_data_imputed)
    return jsonify({'salary': round(prediction[0], 2)})

if __name__ == '__main__':
    app.run(debug=True)
