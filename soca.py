import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize global variable for the dataframe
df = pd.DataFrame()

def load_data(uploaded_file):
    global df
    df = pd.read_csv(uploaded_file)
    df['Test Score'] = pd.to_numeric(df['Test Score'], errors='coerce')

def perform_performance_analysis(df):
    if not pd.api.types.is_numeric_dtype(df['Test Score']):
        st.error("Error: 'Test Score' column must be numeric.")
        return
    
    # Average Test Score by Chapter
    avg_score_by_chapter = df.groupby('Test Chapter')['Test Score'].mean().reset_index()
    avg_score_by_chapter.columns = ['Test Chapter', 'Average Test Score']
    
    st.subheader("Average Test Score by Chapter")
    st.write(avg_score_by_chapter)
    
    # Plot Average Test Score by Chapter
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Test Chapter', y='Average Test Score', data=avg_score_by_chapter, palette='viridis', ax=ax)
    plt.title('Average Test Score by Chapter')
    plt.xlabel('Test Chapter')
    plt.ylabel('Average Test Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Check for empty 'Test Score' data
    if df['Test Score'].dropna().empty:
        st.error("Error: No valid 'Test Score' data available.")
        return
    
    # Score Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    test_scores = df['Test Score'].dropna().values
    ax.hist(test_scores, bins=10, edgecolor='black')
    plt.title('Distribution of Test Scores')
    plt.xlabel('Test Score')
    plt.ylabel('Frequency')
    st.pyplot(fig)

def perform_skills_analysis(df):
    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    df['Strength_encoded'] = label_encoder.fit_transform(df['Strength'].astype(str))
    df['Opportunity_encoded'] = label_encoder.fit_transform(df['Opportunity'].astype(str))
    df['Challenge_encoded'] = label_encoder.fit_transform(df['Challenge'].astype(str))
    
    # Skill Frequency Analysis
    skill_frequency = pd.concat([df['Strength'], df['Opportunity'], df['Challenge']]).value_counts().reset_index()
    skill_frequency.columns = ['Skill', 'Frequency']
    
    st.subheader("Skill Frequency Analysis")
    st.write(skill_frequency)
    
    # Plot Skill Frequency
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Skill', data=skill_frequency, palette='Set2', ax=ax)
    plt.title('Frequency of Skills')
    plt.xlabel('Frequency')
    plt.ylabel('Skill')
    st.pyplot(fig)

def display_correlation(df, chapter, column):
    if df.empty:
        st.error("No data available. Please upload a CSV file.")
        return
    
    if not chapter or not column:
        st.error("Please select both a chapter and a column.")
        return
    
    # Filter data for the selected chapter and create a copy to avoid SettingWithCopyWarning
    df_filtered = df[df['Test Chapter'] == chapter].copy()
    
    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    df_filtered['Strength_encoded'] = label_encoder.fit_transform(df_filtered['Strength'].astype(str))
    df_filtered['Opportunity_encoded'] = label_encoder.fit_transform(df_filtered['Opportunity'].astype(str))
    df_filtered['Challenge_encoded'] = label_encoder.fit_transform(df_filtered['Challenge'].astype(str))

    # Calculate correlation matrix
    correlation_matrix = df_filtered[['Test Score', 'Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']].corr()

    # Display the correlation matrix
    st.subheader(f"Correlation matrix for '{chapter}'")
    st.write(correlation_matrix)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True, square=True, fmt=".2f", ax=ax)
    plt.title(f"Correlation Heatmap for '{chapter}'")
    st.pyplot(fig)
    
    # Display only the correlation between the selected column and 'Test Score'
    if column in correlation_matrix.columns:
        correlation_value = df_filtered['Test Score'].corr(df_filtered[column])
        st.write(f"Correlation between '{column}' and 'Test Score' for '{chapter}': {correlation_value}")

# Streamlit App Layout
st.title('Student Performance Analysis')

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    load_data(uploaded_file)
    st.success("Data loaded successfully.")
    
    # Perform additional analyses
    perform_performance_analysis(df)
    perform_skills_analysis(df)
    
    # Dropdowns for correlation analysis
    chapter = st.selectbox("Select Chapter", options=df['Test Chapter'].unique())
    column = st.selectbox("Select Column to Correlate With", options=['Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded'])
    
    if st.button("Show Correlation"):
        display_correlation(df, chapter, column)
