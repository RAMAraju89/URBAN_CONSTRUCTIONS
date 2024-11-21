import streamlit as st
import requests
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from snowflake.snowpark.functions import col
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Get the current Snowflake session
cnx = st.connection("snowflake")
session =cnx.session()
# Fetch the data from Snowflake
snowflake_df = session.table("URBAN_CONSTRUCTION_DATA")

# Convert Snowpark DataFrame to Pandas DataFrame for use in Streamlit
df = snowflake_df.to_pandas()

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
    model_filename = f"{target_name}_model.pkl.gz"
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
user_inputs = {col: st.number_input(col, min_value=0) for col in X.columns}
input_array = np.array([list(user_inputs.values())])

for target_name, model in models.items():
    if st.button(f"Predict {target_name}"):
        prediction = model.predict(input_array)
        st.write(f"{target_name} Prediction: {prediction[0]}")









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

# # Prepare prediction values
predictions = []

# Collect predictions for each target
for target_name, model in models.items():
    #if st.button(f"Predict {target_name}"):
        prediction = model.predict(input_array)
        predictions.append(prediction[0])  # Append the prediction result to the list
        st.write(f"{target_name} Prediction: {prediction[0]}")

# Plot the predictions if they are available
if predictions:
    plot_predictions(predictions)








# Extract relevant columns for comparison
comparison_columns = {
    "1000+ Population Habitations Covered": [
        "Habitations with 1000+ Population - Habitations Covered (No.s) - Achievement",
        "Habitations with 1000+ Population - Habitations Covered (No.s) - Target"
    ],
    "500+ Population Habitations Covered": [
        "Habitations with 500+ Population - Habitations Covered (No.) - Achievement",
        "Habitations with 500+ Population - Habitations Covered (No.) - Target"
    ],
    "1000+ Population Road Length (KM)": [
        "Habitations with 1000+ Population - Length (KM) - Achievement",
        "Habitations with 1000+ Population - Length (KM) - Target"
    ],
    "500+ Population Road Length (KM)": [
        "Habitations with 500+ Population - Length (KM) - Achievement",
        "Habitations with 500+ Population - Length (KM) - Target"
    ]
}

# Prepare a long-format DataFrame for charting
chart_data = []
for category, cols in comparison_columns.items():
    for row in df.index:
        chart_data.append({
            "Category": category,
            "Type": "Achievement",
            "Value": df.loc[row, cols[0]]
        })
        chart_data.append({
            "Category": category,
            "Type": "Target",
            "Value": df.loc[row, cols[1]]
        })

chart_df = pd.DataFrame(chart_data)

# Altair bar chart comparing achievements vs targets
chart = alt.Chart(chart_df).mark_bar().encode(
    x=alt.X('Category:N', title='Category'),
    y=alt.Y('Value:Q', title='Values'),
    color='Type:N',
    column=alt.Column('Type:N', title='Comparison', spacing=10)
).properties(
    title='Actual vs Target Values for Y1 (Project Approval)',
    width=150,
    height=300
)

# Display the chart
st.subheader("Insight: Y1 (Project Approval) - Actual vs Target Comparison")
st.altair_chart(chart, use_container_width=True)







# Calculate compliance percentage for Y3
total_projects = len(df)
compliant_projects = df['Y3_EHS_Adherence'].sum()
non_compliant_projects = total_projects - compliant_projects

# Calculate percentages
compliance_data = {
    "Category": ["Compliant", "Non-Compliant"],
    "Count": [compliant_projects, non_compliant_projects],
    "Percentage": [
        (compliant_projects / total_projects) * 100,
        (non_compliant_projects / total_projects) * 100
    ]
}

compliance_df = pd.DataFrame(compliance_data)

# Visualization: Pie chart for percentage compliance
st.subheader("Insight: Y3 (EHS Standards) - Compliance Analysis")

st.write(
    "This visualization shows the percentage of projects adhering to environmental, health, and safety (EHS) standards. "
    "Projects that consistently utilize quality materials and undergo state-level inspections tend to comply with these standards."
)

# Option 1: Bar Chart
st.bar_chart(compliance_df.set_index("Category")["Count"], use_container_width=True)

# Option 2: Altair Pie Chart
import altair as alt

pie_chart = alt.Chart(compliance_df).mark_arc(innerRadius=50).encode(
    theta=alt.Theta(field="Count", type="quantitative", title="Projects"),
    color=alt.Color(field="Category", type="nominal", title="Compliance Status"),
    tooltip=["Category", "Count", alt.Tooltip("Percentage:Q", format=".1f")]
).properties(
    title="Y3 (EHS Standards) Compliance Breakdown"
)

# Render the chart
st.altair_chart(pie_chart, use_container_width=True)



