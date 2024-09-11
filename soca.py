import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Initialize global variable for the dataframe
df = pd.DataFrame()

# File upload using Streamlit's built-in widget
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Function to load and process the uploaded file
def load_data():
    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Ensure 'Test Score' is numeric
        df['Test Score'] = pd.to_numeric(df['Test Score'], errors='coerce')

        # Encode categorical columns using LabelEncoder
        label_encoder = LabelEncoder()
        df['Strength_encoded'] = label_encoder.fit_transform(df['Strength'].astype(str))
        df['Opportunity_encoded'] = label_encoder.fit_transform(df['Opportunity'].astype(str))
        df['Challenge_encoded'] = label_encoder.fit_transform(df['Challenge'].astype(str))

        return df
    return None

df = load_data()

# Dropdown selections for different analyses
if df is not None:
    available_chapters = df['Test Chapter'].unique()
    
    # Performance Analysis
    st.sidebar.header("Performance Analysis")
    selected_chapter_perf = st.sidebar.selectbox('Select Performance Chapter', available_chapters)
    
    # Skills Analysis
    st.sidebar.header("Skills Analysis")
    selected_chapter_skills = st.sidebar.selectbox('Select Skills Chapter', available_chapters)
    
    # Correlation Heatmap
    st.sidebar.header("Correlation Heatmap")
    selected_chapter_corr = st.sidebar.selectbox('Select Correlation Chapter', available_chapters)
    correlation_columns = ['Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']
    selected_column_corr = st.sidebar.selectbox('Select Column to Correlate With', correlation_columns)

    # Performance Analysis
    if st.sidebar.button('Plot Performance Analysis'):
        avg_score_by_chapter = df.groupby('Test Chapter')['Test Score'].mean().reset_index()
        avg_score_by_chapter.columns = ['Test Chapter', 'Average Test Score']

        st.write(f"Performance Analysis for '{selected_chapter_perf}'")
        st.dataframe(avg_score_by_chapter[avg_score_by_chapter['Test Chapter'] == selected_chapter_perf])
        
        # Plot Average Test Score by Chapter
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Test Chapter', y='Average Test Score', data=avg_score_by_chapter, palette='viridis', ax=ax)
        ax.set_title(f'Average Test Score for {selected_chapter_perf}')
        st.pyplot(fig)

    # Skills Analysis
    if st.sidebar.button('Plot Skills Analysis'):
        skill_frequency = pd.concat([df['Strength'], df['Opportunity'], df['Challenge']]).value_counts().reset_index()
        skill_frequency.columns = ['Skill', 'Frequency']
        
        st.write(f"Skills Analysis for '{selected_chapter_skills}'")
        st.dataframe(skill_frequency)
        
        # Plot Skill Frequency
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='Frequency', y='Skill', data=skill_frequency, palette='Set2', ax=ax)
        ax.set_title(f'Frequency of Skills for {selected_chapter_skills}')
        st.pyplot(fig)

    # Correlation Heatmap
    if st.sidebar.button('Plot Correlation Heatmap'):
        # Filter data for the selected chapter
        df_filtered = df[df['Test Chapter'] == selected_chapter_corr].copy()
        
        # Calculate correlation matrix
        correlation_matrix = df_filtered[['Test Score', 'Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']].corr()

        st.write(f"Correlation matrix for '{selected_chapter_corr}'")
        st.dataframe(correlation_matrix)
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True, square=True, fmt=".2f", ax=ax)
        ax.set_title(f"Correlation Heatmap for '{selected_chapter_corr}'")
        st.pyplot(fig)

    # Dynamic Chapter-based Analysis
    st.sidebar.header("Chapter Analysis")
    selected_chapter_dynamic = st.sidebar.selectbox('Select Chapter for Dynamic Analysis', available_chapters)

    if selected_chapter_dynamic:
        filtered_df = df[df['Test Chapter'] == selected_chapter_dynamic]
        
        # Display statistics
        avg_score = filtered_df['Test Score'].mean()
        num_entries = len(filtered_df)
        st.write(f"**Statistics for {selected_chapter_dynamic}:**")
        st.write(f"Average Test Score: {avg_score:.2f}")
        st.write(f"Number of Entries: {num_entries}")
        
        # Plot score distribution by Strength, Opportunity, Challenge
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        filtered_df.groupby('Strength')['Test Score'].mean().plot(kind='bar', ax=axes[0], title='Score Distribution by Strength', color='skyblue')
        filtered_df.groupby('Opportunity')['Test Score'].mean().plot(kind='bar', ax=axes[1], title='Score Distribution by Opportunity', color='lightgreen')
        filtered_df.groupby('Challenge')['Test Score'].mean().plot(kind='bar', ax=axes[2], title='Score Distribution by Challenge', color='lightcoral')

        st.pyplot(fig)
else:
    st.write("Please upload a CSV file to start analysis.")
