import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Streamlit app title
st.title("Student Test Performance Analysis")

# Initialize global variable for the dataframe
df = pd.DataFrame()

# Function to update dropdown options
def update_dropdowns():
    global df
    if 'Test Chapter' in df.columns:
        available_chapters = df['Test Chapter'].unique()
        correlation_columns = ['Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']
        return available_chapters, correlation_columns
    else:
        return [], []

# Function to display statistics and plots
def update_analysis(chapter):
    if chapter and 'Test Chapter' in df.columns:
        filtered_df = df[df['Test Chapter'] == chapter]
        avg_score = filtered_df['Test Score'].mean()
        num_entries = len(filtered_df)
        
        st.write(f"**Statistics for {chapter}:**")
        st.write(f"Average Test Score: {avg_score:.2f}")
        st.write(f"Number of Entries: {num_entries}")
        
        # Plot score distribution by Strength, Opportunity, Challenge
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        strength_scores = filtered_df.groupby('Strength')['Test Score'].mean()
        axs[0].bar(strength_scores.index, strength_scores, color='skyblue')
        axs[0].set_title('Score Distribution by Strength')
        axs[0].set_xticklabels(strength_scores.index, rotation=45)
        
        opportunity_scores = filtered_df.groupby('Opportunity')['Test Score'].mean()
        axs[1].bar(opportunity_scores.index, opportunity_scores, color='lightgreen')
        axs[1].set_title('Score Distribution by Opportunity')
        axs[1].set_xticklabels(opportunity_scores.index, rotation=45)
        
        challenge_scores = filtered_df.groupby('Challenge')['Test Score'].mean()
        axs[2].bar(challenge_scores.index, challenge_scores, color='lightcoral')
        axs[2].set_title('Score Distribution by Challenge')
        axs[2].set_xticklabels(challenge_scores.index, rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Please select a valid chapter.")

# Function to handle file upload and update dataframe
def handle_upload(uploaded_file):
    global df
    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Ensure 'Test Score' is numeric
        df['Test Score'] = pd.to_numeric(df['Test Score'], errors='coerce')
        
        # Encode categorical columns for correlation
        label_encoder = LabelEncoder()
        df['Strength_encoded'] = label_encoder.fit_transform(df['Strength'].astype(str))
        df['Opportunity_encoded'] = label_encoder.fit_transform(df['Opportunity'].astype(str))
        df['Challenge_encoded'] = label_encoder.fit_transform(df['Challenge'].astype(str))
        
        st.success("Data loaded successfully!")
        return True
    return False

# File upload widget
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# Load data if a file is uploaded
if uploaded_file is not None:
    if handle_upload(uploaded_file):
        available_chapters, correlation_columns = update_dropdowns()
        
        # Performance Analysis
        st.subheader("Performance Analysis")
        chapter_perf = st.selectbox("Select Performance Chapter", available_chapters)
        if st.button("Plot Performance Analysis"):
            update_analysis(chapter_perf)
        
        # Skills Analysis
        st.subheader("Skills Analysis")
        chapter_skills = st.selectbox("Select Skills Chapter", available_chapters)
        if st.button("Plot Skills Analysis"):
            update_analysis(chapter_skills)
        
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        chapter_corr = st.selectbox("Select Correlation Chapter", available_chapters)
        column_corr = st.selectbox("Correlate with", correlation_columns)
        if st.button("Plot Correlation Heatmap"):
            if chapter_corr and column_corr:
                filtered_df = df[df['Test Chapter'] == chapter_corr]
                correlation_matrix = filtered_df[['Test Score', 'Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']].corr()
                
                st.write(f"Correlation matrix for {chapter_corr}:")
                st.write(correlation_matrix)
                
                # Plot heatmap
                fig, ax = plt.subplots()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True, square=True, fmt=".2f", ax=ax)
                st.pyplot(fig)
