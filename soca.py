import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from io import StringIO

# Set page config
st.set_page_config(page_title="Data Analysis App", layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Performance Analysis", "Skills Analysis", "Correlation Analysis"])

# File upload
def handle_upload():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Test Score'] = pd.to_numeric(df['Test Score'], errors='coerce')
        st.session_state.df = df
        st.success("Data loaded successfully!")
        st.write(df.head())

# Performance Analysis
def performance_analysis():
    if st.session_state.df.empty:
        st.warning("Please upload data first.")
        return

    chapter = st.selectbox("Select Test Chapter", options=st.session_state.df['Test Chapter'].unique())
    
    avg_score_by_chapter = st.session_state.df.groupby('Test Chapter')['Test Score'].mean().reset_index()
    
    st.write(f"Performance Analysis for '{chapter}':")
    st.write(avg_score_by_chapter[avg_score_by_chapter['Test Chapter'] == chapter])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Test Chapter', y='Average Test Score', data=avg_score_by_chapter, palette='viridis', ax=ax)
    plt.title(f'Average Test Score by Chapter')
    plt.xlabel('Test Chapter')
    plt.ylabel('Average Test Score')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Skills Analysis
def skills_analysis():
    if st.session_state.df.empty:
        st.warning("Please upload data first.")
        return

    chapter = st.selectbox("Select Test Chapter", options=st.session_state.df['Test Chapter'].unique())
    
    label_encoder = LabelEncoder()
    df = st.session_state.df.copy()
    df['Strength_encoded'] = label_encoder.fit_transform(df['Strength'].astype(str))
    df['Opportunity_encoded'] = label_encoder.fit_transform(df['Opportunity'].astype(str))
    df['Challenge_encoded'] = label_encoder.fit_transform(df['Challenge'].astype(str))

    skill_frequency = pd.concat([df['Strength'], df['Opportunity'], df['Challenge']]).value_counts().reset_index()
    skill_frequency.columns = ['Skill', 'Frequency']
    
    st.write(f"Skills Analysis for '{chapter}':")
    st.write(skill_frequency)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Skill', data=skill_frequency, palette='Set2', ax=ax)
    plt.title(f'Frequency of Skills')
    plt.xlabel('Frequency')
    plt.ylabel('Skill')
    st.pyplot(fig)

# Correlation Analysis
def correlation_analysis():
    if st.session_state.df.empty:
        st.warning("Please upload data first.")
        return

    chapter = st.selectbox("Select Test Chapter", options=st.session_state.df['Test Chapter'].unique())
    column = st.selectbox("Correlate with", options=['Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded'])
    
    df_filtered = st.session_state.df[st.session_state.df['Test Chapter'] == chapter].copy()
    
    label_encoder = LabelEncoder()
    df_filtered['Strength_encoded'] = label_encoder.fit_transform(df_filtered['Strength'].astype(str))
    df_filtered['Opportunity_encoded'] = label_encoder.fit_transform(df_filtered['Opportunity'].astype(str))
    df_filtered['Challenge_encoded'] = label_encoder.fit_transform(df_filtered['Challenge'].astype(str))

    correlation_matrix = df_filtered[['Test Score', 'Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']].corr()

    st.write(f"Correlation matrix for '{chapter}':")
    st.write(correlation_matrix)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True, square=True, fmt=".2f", ax=ax)
    plt.title(f"Correlation Heatmap for '{chapter}'")
    st.pyplot(fig)

# Main app logic
def main():
    if page == "Upload Data":
        st.title("Upload Data")
        handle_upload()
    elif page == "Performance Analysis":
        st.title("Performance Analysis")
        performance_analysis()
    elif page == "Skills Analysis":
        st.title("Skills Analysis")
        skills_analysis()
    elif page == "Correlation Analysis":
        st.title("Correlation Analysis")
        correlation_analysis()

if __name__ == "__main__":
    main()
