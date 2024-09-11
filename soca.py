import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import io

# Initialize global variable for the dataframe
df = pd.DataFrame()

# File uploader widget
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Dropdowns and buttons
chapter_dropdown_perf = st.selectbox('Performance Chapter:', [])
chapter_dropdown_skills = st.selectbox('Skills Chapter:', [])
chapter_dropdown_corr = st.selectbox('Correlation Chapter:', [])
column_dropdown_corr = st.selectbox('Correlate with:', ['Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded'])

if uploaded_file:
    # Read the uploaded file into a DataFrame
    df = pd.read_csv(uploaded_file)

    # Ensure 'Test Score' is numeric
    df['Test Score'] = pd.to_numeric(df['Test Score'], errors='coerce')
    
    # Update dropdown options
    available_chapters = df['Test Chapter'].unique()
    chapter_dropdown_perf = st.selectbox('Performance Chapter:', available_chapters)
    chapter_dropdown_skills = st.selectbox('Skills Chapter:', available_chapters)
    chapter_dropdown_corr = st.selectbox('Correlation Chapter:', available_chapters)
    chapter_selector = st.selectbox('Test Chapter:', available_chapters)

    st.write("Data loaded successfully. Please select options from the dropdowns.")

    # Encode categorical columns for correlation analysis
    label_encoder = LabelEncoder()
    df['Strength_encoded'] = label_encoder.fit_transform(df['Strength'].astype(str))
    df['Opportunity_encoded'] = label_encoder.fit_transform(df['Opportunity'].astype(str))
    df['Challenge_encoded'] = label_encoder.fit_transform(df['Challenge'].astype(str))

    def perform_performance_analysis(chapter):
        if chapter:
            # Average Test Score by Chapter
            avg_score_by_chapter = df.groupby('Test Chapter')['Test Score'].mean().reset_index()
            avg_score_by_chapter.columns = ['Test Chapter', 'Average Test Score']
            
            st.write(f"\nPerformance Analysis for '{chapter}':")
            st.write(avg_score_by_chapter[avg_score_by_chapter['Test Chapter'] == chapter])
            
            # Plot Average Test Score by Chapter
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Test Chapter', y='Average Test Score', data=avg_score_by_chapter, palette='viridis')
            plt.title(f'Average Test Score for {chapter}')
            plt.xlabel('Test Chapter')
            plt.ylabel('Average Test Score')
            plt.xticks(rotation=45)
            st.pyplot()

    def perform_skills_analysis(chapter):
        if chapter:
            # Skill Frequency Analysis
            skill_frequency = pd.concat([df['Strength'], df['Opportunity'], df['Challenge']]).value_counts().reset_index()
            skill_frequency.columns = ['Skill', 'Frequency']
            
            st.write(f"\nSkills Analysis for '{chapter}':")
            st.write(skill_frequency)
            
            # Plot Skill Frequency
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Frequency', y='Skill', data=skill_frequency, palette='Set2')
            plt.title(f'Frequency of Skills for {chapter}')
            plt.xlabel('Frequency')
            plt.ylabel('Skill')
            st.pyplot()

    def display_correlation(chapter, column):
        if chapter and column:
            # Filter data for the selected chapter
            df_filtered = df[df['Test Chapter'] == chapter].copy()

            # Calculate correlation matrix
            correlation_matrix = df_filtered[['Test Score', 'Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']].corr()

            # Display the correlation matrix
            st.write(f"Correlation matrix for '{chapter}':")
            st.write(correlation_matrix)
            
            # Plot heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True, square=True, fmt=".2f")
            plt.title(f"Correlation Heatmap for '{chapter}'")
            st.pyplot()

    def update_analysis(chapter):
        if 'Test Chapter' in df.columns:
            # Filter data based on selected chapter
            filtered_df = df[df['Test Chapter'] == chapter]
            
            # Display statistics
            avg_score = filtered_df['Test Score'].mean()
            num_entries = len(filtered_df)
            st.write(f"**Statistics for {chapter}:**")
            st.write(f"Average Test Score: {avg_score:.2f}")
            st.write(f"Number of Entries: {num_entries}")
            
            # Plot score distribution by Strength
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))
            strength_scores = filtered_df.groupby('Strength')['Test Score'].mean()
            strength_scores.plot(kind='bar', ax=axs[0], title='Score Distribution by Strength', color='skyblue')
            
            # Plot score distribution by Opportunity
            opportunity_scores = filtered_df.groupby('Opportunity')['Test Score'].mean()
            opportunity_scores.plot(kind='bar', ax=axs[1], title='Score Distribution by Opportunity', color='lightgreen')
            
            # Plot score distribution by Challenge
            challenge_scores = filtered_df.groupby('Challenge')['Test Score'].mean()
            challenge_scores.plot(kind='bar', ax=axs[2], title='Score Distribution by Challenge', color='lightcoral')
            
            plt.tight_layout()
            st.pyplot(fig)

    # Add interactive elements for analysis
    if chapter_dropdown_perf:
        perform_performance_analysis(chapter_dropdown_perf)
    if chapter_dropdown_skills:
        perform_skills_analysis(chapter_dropdown_skills)
    if chapter_dropdown_corr and column_dropdown_corr:
        display_correlation(chapter_dropdown_corr, column_dropdown_corr)
    if chapter_selector:
        update_analysis(chapter_selector)
