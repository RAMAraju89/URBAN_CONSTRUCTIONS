import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import base64
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

conn = st.connection("snowflake")

@st.cache_data
def load_table():
    session = conn.session()
    return session.table("URBAN_CONSTRUCTION_DB.CONSTRUCTION_PROJECTS.URBAN_CONSTRUCTION_DATA").to_pandas()

df = load_table()

# Display the data in Streamlit
st.title("Urban Construction Safety and Compliance Dashboard")
st.write(
    """This dashboard visualizes safety compliance, recycling efforts, 
    and sustainability practices in urban construction projects across major cities in India."""
)

# Handle missing data
df.fillna(0, inplace=True)

# Remove the last two rows
df = df.iloc[:-2]

# Create synthetic data
num_synthetic_samples = 100
synthetic_data = df.sample(num_synthetic_samples, replace=True).copy()
numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns

# Add random noise to each numeric column
for col in numeric_cols:
    noise = np.random.normal(0, 0.01 * df[col].std(), num_synthetic_samples)
    synthetic_data[col] += noise

# Combine original and synthetic data
df = pd.concat([df, synthetic_data], ignore_index=True)


# Project Approval (Y1): 1 if Achievements >= Targets, else 0
df['Y1_Project_Approval'] = (
    (df['Habitations with 1000+ Population - Habitations Covered (No.s) - Achievement'] >= df['Habitations with 1000+ Population - Habitations Covered (No.s) - Target']) &
    (df['Habitations with 500+ Population - Habitations Covered (No.) - Achievement'] >= df['Habitations with 500+ Population - Habitations Covered (No.) - Target']) &
    (df['Habitations with 1000+ Population - Length (KM) - Achievement'] >= df['Habitations with 1000+ Population - Length (KM) - Target']) &
    (df['Habitations with 500+ Population - Length (KM) - Achievement'] >= df['Habitations with 500+ Population - Length (KM) - Target'])
).astype(int)

# Safety Practices Compliance (Y2): 1 if sufficient inspections and satisfactory quality observed
df['Y2_Safety_Compliance'] = (
    (df['Quality Observed in Both Years by State Level Monitors: Good'] > 0) |
    (df['Quality Observed in Both Years by State Level Monitors: Satisfactory'] > 0)
).astype(int)

# EHS Standards Adherence (Y3): 1 if utilized quality materials and had state-level inspections
df['Y3_EHS_Adherence'] = (
    (df['Utilised Tested and Quality Materials'] > df['Not utilised Tested and Quality Materials']) &
    (df['No. of Inspections conducted by State Level Monitors (SQM) Independent of executive Agency deployed under Bharat Nirman Programme as reported by: State Level Authorities: 2005-06'] > 0)
).astype(int)

# Waste Recycling Practices (Y4): 1 if uses quality materials consistently
df['Y4_Recycling_Practices'] = (
    (df['Utilised Tested and Quality Materials'] > df['Not utilised Tested and Quality Materials']) &
    (df['Quality and Quantity of Materials used by the Contractor - Used substandard Quality Cement/ Bricks instead of Stones'] == 0)
).astype(int)

# Eco-Friendly Material Usage (Y5): 1 if high-quality material usage and no substandard practices
df['Y5_Eco_Friendly_Materials'] = (
    (df['Utilised Tested and Quality Materials'] > df['Not utilised Tested and Quality Materials']) &
    (df['Quality and Quantity of Materials used by the Contractor - Not up the Standard/lack of Thickness'] == 0) &
    (df['Quality and Quantity of Materials used by the Contractor - Used Inadequate Quantity of Black Tapping'] == 0)
).astype(int)

# Split the data into features and target labels
X = df.drop(columns=['Y1_Project_Approval', 'Y2_Safety_Compliance', 'Y3_EHS_Adherence', 'Y4_Recycling_Practices', 'Y5_Eco_Friendly_Materials']).select_dtypes(include=[np.number])
Y1 = df['Y1_Project_Approval']
Y2 = df['Y2_Safety_Compliance']
Y3 = df['Y3_EHS_Adherence']
Y4 = df['Y4_Recycling_Practices']
Y5 = df['Y5_Eco_Friendly_Materials']

# Dictionary to store models
models = {}

# Train and save a Random Forest model for each target
for target_name, Y in {'Y1': Y1, 'Y2': Y2, 'Y3': Y3, 'Y4': Y4, 'Y5': Y5}.items():
    # Split the data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier()
    model.fit(X_train, Y_train)

    # Save the model
    model_filename = f"{target_name}_model.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    models[target_name] = model  # Store model in dictionary

# Function to upload a model to Snowflake stage
def upload_to_snowflake(local_filepath, snowflake_stage='@"URBAN_CONSTRUCTION_DB"."CONSTRUCTION_PROJECTS"."URBAN_CONSTRUCTION_PREDICTIVE_MODEL"'):
    try:
        with open(local_filepath, 'rb') as file:
            session.file.put_stream(file, f"{snowflake_stage}/{local_filepath.split('/')[-1]}", auto_compress=True)
    except Exception as e:
        st.error(f"Error uploading to Snowflake: {e}")

# Load models from Snowflake and prepare for predictions
def load_model_from_snowflake(session, stage_path, local_file_name):
    try:
        session.file.get(stage_path, f"./{local_file_name}")
        with open(local_file_name, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model from {stage_path}: {e}")
        return None

# Load models into memory
loaded_models = {}
model_paths = {
    'Y1': '@"URBAN_CONSTRUCTION_DB"."CONSTRUCTION_PROJECTS"."URBAN_CONSTRUCTION_PREDICTIVE_MODEL"/Y1_model.pkl',
    'Y2': '@"URBAN_CONSTRUCTION_DB"."CONSTRUCTION_PROJECTS"."URBAN_CONSTRUCTION_PREDICTIVE_MODEL"/Y2_model.pkl',
    'Y3': '@"URBAN_CONSTRUCTION_DB"."CONSTRUCTION_PROJECTS"."URBAN_CONSTRUCTION_PREDICTIVE_MODEL"/Y3_model.pkl',
    'Y4': '@"URBAN_CONSTRUCTION_DB"."CONSTRUCTION_PROJECTS"."URBAN_CONSTRUCTION_PREDICTIVE_MODEL"/Y4_model.pkl',
    'Y5': '@"URBAN_CONSTRUCTION_DB"."CONSTRUCTION_PROJECTS"."URBAN_CONSTRUCTION_PREDICTIVE_MODEL"/Y5_model.pkl'
}

# Prepare for predictions when button is pressed
user_inputs = {col: st.number_input(col, min_value=0.0) for col in X.columns}
input_array = np.array([list(user_inputs.values())])

# Add condition to predict only when button is pressed
if st.button("Predict All"):
    predictions = {}
    for target_name, model in models.items():
        predictions[target_name] = model.predict(input_array)[0]
        
    # Function to plot predictions
    def plot_predictions(predictions):
        labels = ['Y1_Project_Approval', 'Y2_Safety_Compliance', 'Y3_EHS_Adherence', 'Y4_Recycling_Practices', 'Y5_Eco_Friendly_Materials']
        plt.figure(figsize=(15, 6))
        sns.barplot(x=labels, y=predictions, palette='Blues_d')
        plt.title('Predictions for Safety and Compliance Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Prediction Value')
        plt.ylim(0, 1)
        st.pyplot(plt)

    # Collect predictions for each target
    predictions_values = []
    labels = ['Y1_Project_Approval', 'Y2_Safety_Compliance', 'Y3_EHS_Adherence', 'Y4_Recycling_Practices', 'Y5_Eco_Friendly_Materials']
    for i, (target_name, prediction) in enumerate(predictions.items()):
        predictions_values.append(prediction)
        st.write(f"{labels[i]} Prediction: {prediction}")
        
    # Plot the predictions if they are available
    if predictions_values:
        plot_predictions(predictions_values)












# Function to download file from Snowflake stage
def download_image_from_snowflake(session, stage_path, local_filename):
    try:
        # Retrieve the file from the Snowflake stage
        session.file.get(stage_path, f"./{local_filename}")
        st.success(f"File downloaded successfully: {local_filename}")
        return local_filename
    except Exception as e:
        st.error(f"Error downloading image from Snowflake: {e}")
        return None

# Function to set background image
def set_background_image(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded_image = base64.b64encode(file.read()).decode()
        css = f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpg;base64,{encoded_image}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error setting background image: {e}")

# Streamlit App
def main():
    # Establish connection to Snowflake
    conn = st.connection("snowflake")
    session = conn.session()

    # Define Snowflake stage path for the background image
    snowflake_stage_path = '@"URBAN_CONSTRUCTION_DB"."CONSTRUCTION_PROJECTS"."BGI"/urban-construction bgi.jpg'
    local_filename = "urban-construction bgi.jpg"

    # Download the image from Snowflake stage
    image_file = download_image_from_snowflake(session, snowflake_stage_path, local_filename)

    # Set the background image if download was successful
    if image_file:
        set_background_image(image_file)

    # App content
    st.title("Welcome to the Construction Dashboard")
    st.write("This app showcases a background image in Streamlit.")
    st.button("Click Me")

if __name__ == "__main__":
    main()


