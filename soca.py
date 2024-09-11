import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import io

# Initialize global variable for the dataframe
df = pd.DataFrame()

def handle_upload(uploaded_file):
    global df
    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        file_content = uploaded_file.read()
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        
        # Update dropdown options
        update_dropdowns()
        st.write("Data loaded successfully. Please select options from the dropdowns.")

def update_dropdowns():
    global chapter_options, column_options
    
    # Update chapter dropdown options
    available_chapters = df['Test Chapter'].unique()
    chapter_options = available_chapters
    
    # Update column dropdown options
    correlation_columns = ['Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']
    column_options = correlation_columns

def display_correlation(chapter, column):
    if df.empty:
        st.write("No data available. Please upload a CSV file.")
        return
    
    if not chapter or not column:
        st.write("Please select both a chapter and a column.")
        return
    
    # Filter data for the selected chapter and create a copy to avoid SettingWithCopyWarning
    df_filtered = df[df['Test Chapter'] == chapter].copy()
    
    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    df_filtered['Strength_encoded'] = label_encoder.fit_transform(df_filtered['Strength'])
    df_filtered['Opportunity_encoded'] = label_encoder.fit_transform(df_filtered['Opportunity'])
    df_filtered['Challenge_encoded'] = label_encoder.fit_transform(df_filtered['Challenge'])

    # Calculate correlation matrix
    correlation_matrix = df_filtered[['Test Score', 'Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']].corr()

    # Display the correlation matrix
    st.write(f"Correlation matrix for '{chapter}':")
    st.write(correlation_matrix)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True, square=True, fmt=".2f")
    plt.title(f"Correlation Heatmap for '{chapter}'")
    st.pyplot(plt)
    
    # Display only the correlation between the selected column and 'Test Score'
    if column in correlation_matrix.columns:
        correlation_value = df_filtered['Test Score'].corr(df_filtered[column])
        st.write(f"Correlation between '{column}' and 'Test Score' for '{chapter}': {correlation_value}")

# Streamlit app
st.title("Correlation Analysis Tool")

# File uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# Handle file upload
if uploaded_file:
    handle_upload(uploaded_file)

# Dropdown widgets
if not df.empty:
    chapter = st.selectbox("Select Chapter:", options=chapter_options)
    column = st.selectbox("Select Column to Correlate with Test Score:", options=column_options)
    
    # Display correlation and heatmap
    display_correlation(chapter, column)
