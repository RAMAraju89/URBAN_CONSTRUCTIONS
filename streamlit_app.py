import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from snowflake.snowpark.context import Session
from snowflake.snowpark.exceptions import SnowflakeSQLException

# Initialize connection to Snowflake
@st.cache_resource
def initialize_connection():
    # Assuming Streamlit connection configuration is properly set up
    return st.connection("snowflake").session()

# Load table data
@st.cache_data
def load_table():
    try:
        session = initialize_connection()
        # Specify schema and table as required
        snowflake_table = session.table("URBAN_CONSTRUCTION_DB.CONSTRUCTION_PROJECTS.URBAN_CONSTRUCTION_DATA")
        return snowflake_table.to_pandas()
    except SnowflakeSQLException as e:
        st.error(f"Error loading data from Snowflake: {e}")
        return pd.DataFrame()

# Upload models to Snowflake
def upload_to_snowflake(local_file_path, snowflake_stage):
    session = initialize_connection()
    try:
        with open(local_file_path, "rb") as file:
            # Adjust path for Snowflake stage
            session.file.put_stream(file, f"{snowflake_stage}/{os.path.basename(local_file_path)}", auto_compress=True)
        st.success(f"File {os.path.basename(local_file_path)} uploaded to Snowflake stage {snowflake_stage}.")
    except Exception as e:
        st.error(f"Error uploading file to Snowflake: {e}")

# Load models from Snowflake
def load_model_from_snowflake(local_file_path, snowflake_stage):
    session = initialize_connection()
    try:
        file_path = f"{snowflake_stage}/{os.path.basename(local_file_path)}"
        session.file.get(file_path, local_file_path)
        with open(local_file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model from Snowflake: {e}")
        return None

# Plot predictions
def plot_predictions(predictions):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(predictions)), predictions, color="skyblue")
    ax.set_title("Prediction Results")
    ax.set_xlabel("Target Variable")
    ax.set_ylabel("Prediction Value")
    st.pyplot(fig)

# Main App Logic
def main():
    st.title("Urban Construction Prediction App")
    
    # Load data
    df = load_table()
    if df.empty:
        st.warning("No data available. Ensure Snowflake table is populated.")
        return

    st.write("Data loaded successfully. Displaying first 5 rows:")
    st.dataframe(df.head())

    # Handle missing data
    df.fillna(0, inplace=True)

    # Calculate Project Approval (Y1)
    df['Y1_Project_Approval'] = (
        (df['Habitations with 1000+ Population - Habitations Covered (No.s) - Achievement'] >= df['Habitations with 1000+ Population - Habitations Covered (No.s) - Target']) &
        (df['Habitations with 500+ Population - Habitations Covered (No.) - Achievement'] >= df['Habitations with 500+ Population - Habitations Covered (No.) - Target']) &
        (df['Habitations with 1000+ Population - Length (KM) - Achievement'] >= df['Habitations with 1000+ Population - Length (KM) - Target']) &
        (df['Habitations with 500+ Population - Length (KM) - Achievement'] >= df['Habitations with 500+ Population - Length (KM) - Target'])
    ).astype(int)

    # Calculate Safety Practices Compliance (Y2)
    df['Y2_Safety_Compliance'] = (
        (df['Quality Observed in Both Years by State Level Monitors: Good'] > 0) |
        (df['Quality Observed in Both Years by State Level Monitors: Satisfactory'] > 0)
    ).astype(int)

    # Calculate EHS Standards Adherence (Y3)
    df['Y3_EHS_Adherence'] = (
        (df['Utilised Tested and Quality Materials'] > df['Not utilised Tested and Quality Materials']) &
        (df['No. of Inspections conducted by State Level Monitors (SQM) Independent of executive Agency deployed under Bharat Nirman Programme as reported by: State Level Authorities: 2005-06'] > 0)
    ).astype(int)

    # Calculate Waste Recycling Practices (Y4)
    df['Y4_Recycling_Practices'] = (
        (df['Utilised Tested and Quality Materials'] > df['Not utilised Tested and Quality Materials']) &
        (df['Quality and Quantity of Materials used by the Contractor - Used substandard Quality Cement/ Bricks instead of Stones'] == 0)
    ).astype(int)

    # Calculate Eco-Friendly Material Usage (Y5)
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
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        models[target_name] = model

        # Save the trained model
        model_file_path = f"{target_name}_model.pkl"
        with open(model_file_path, "wb") as f:
            pickle.dump(model, f)
        st.write(f"Model for {target_name} saved locally.")
        upload_to_snowflake(model_file_path, "my_snowflake_stage")

    st.success("Models trained and saved successfully!")

    # Prediction Input
    st.subheader("Prediction Inputs")
    user_inputs = {
        col: st.number_input(col, value=float(df[col].mean())) for col in df.select_dtypes(include=[np.number]).columns
    }
    input_array = np.array([list(user_inputs.values())])

    # Make predictions
    if st.button("Predict All"):
        predictions = {target: model.predict(input_array)[0] for target, model in models.items()}
        st.write("Prediction Results:", predictions)
        plot_predictions(list(predictions.values()))

# Run the app
if __name__ == "__main__":
    main()
