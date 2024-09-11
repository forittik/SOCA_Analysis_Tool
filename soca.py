import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import ipywidgets as widgets
from IPython.display import display, clear_output
from io import StringIO

# Initialize global variable for the dataframe
df = pd.DataFrame()

# Create a file upload widget
upload_widget = widgets.FileUpload(
    accept='.csv',  # Accept only CSV files
    multiple=False  # Single file upload
)

# Create dropdown widgets
chapter_dropdown = widgets.Dropdown(
    options=[],  # Options will be updated dynamically
    value=None,  # Default value will be updated dynamically
    description='Chapter:',
    disabled=False,
)

column_dropdown = widgets.Dropdown(
    options=[],  # Options will be updated dynamically
    description='Correlate with:',
    disabled=False,
)

def handle_upload(change):
    global df
    if upload_widget.value:
        # Read the uploaded file into a DataFrame
        uploaded_file = next(iter(upload_widget.value.values()))
        file_content = uploaded_file['content']
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        
        # Ensure 'Test Score' is numeric
        df['Test Score'] = pd.to_numeric(df['Test Score'], errors='coerce')
        
        # Update dropdown options
        update_dropdowns()
        print("Data loaded successfully. Please select options from the dropdowns.")

        # Perform additional analyses
        perform_performance_analysis()
        perform_skills_analysis()

def update_dropdowns():
    global chapter_dropdown, column_dropdown
    
    # Update chapter dropdown options
    available_chapters = df['Test Chapter'].unique()
    chapter_dropdown.options = available_chapters
    chapter_dropdown.value = available_chapters[0] if available_chapters.size > 0 else None
    
    # Update column dropdown options
    correlation_columns = ['Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']
    column_dropdown.options = correlation_columns
    column_dropdown.value = correlation_columns[0] if correlation_columns else None





def perform_performance_analysis():
    global df
    # Check if 'Test Score' is numeric
    if not pd.api.types.is_numeric_dtype(df['Test Score']):
        print("Error: 'Test Score' column must be numeric.")
        return
    
    # Average Test Score by Chapter
    avg_score_by_chapter = df.groupby('Test Chapter')['Test Score'].mean().reset_index()
    avg_score_by_chapter.columns = ['Test Chapter', 'Average Test Score']
    
    print("\nAverage Test Score by Chapter:")
    print(avg_score_by_chapter)
    
    # Plot Average Test Score by Chapter
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Test Chapter', y='Average Test Score', data=avg_score_by_chapter, palette='viridis')
    plt.title('Average Test Score by Chapter')
    plt.xlabel('Test Chapter')
    plt.ylabel('Average Test Score')
    plt.xticks(rotation=45)
    plt.show()
    
    # Check for empty 'Test Score' data
    if df['Test Score'].dropna().empty:
        print("Error: No valid 'Test Score' data available.")
        return
    
    # Score Distribution
    plt.figure(figsize=(10, 6))
    test_scores = df['Test Score'].dropna().values
    plt.hist(test_scores, bins=10, edgecolor='black')
    plt.title('Distribution of Test Scores')
    plt.xlabel('Test Score')
    plt.ylabel('Frequency')
    plt.show()

# ... (rest of the code remains the same)
def perform_skills_analysis():
    global df
    # Encode categorical columns using LabelEncoder
    label_encoder = LabelEncoder()
    df['Strength_encoded'] = label_encoder.fit_transform(df['Strength'].astype(str))
    df['Opportunity_encoded'] = label_encoder.fit_transform(df['Opportunity'].astype(str))
    df['Challenge_encoded'] = label_encoder.fit_transform(df['Challenge'].astype(str))
    
    # Skill Frequency Analysis
    skill_frequency = pd.concat([df['Strength'], df['Opportunity'], df['Challenge']]).value_counts().reset_index()
    skill_frequency.columns = ['Skill', 'Frequency']
    
    print("\nSkill Frequency Analysis:")
    print(skill_frequency)
    
    # Plot Skill Frequency
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Skill', data=skill_frequency, palette='Set2')
    plt.title('Frequency of Skills')
    plt.xlabel('Frequency')
    plt.ylabel('Skill')
    plt.show()

def display_correlation(change):
    clear_output(wait=True)  # Clear previous outputs
    display(upload_widget, chapter_dropdown, column_dropdown)  # Display widgets again
    
    if df.empty:
        print("No data available. Please upload a CSV file.")
        return
    
    chapter = chapter_dropdown.value
    column = column_dropdown.value
    
    if not chapter or not column:
        print("Please select both a chapter and a column.")
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
    print(f"Correlation matrix for '{chapter}':\n", correlation_matrix)
    
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True, square=True, fmt=".2f")
    plt.title(f"Correlation Heatmap for '{chapter}'")
    plt.show()
    
    # Display only the correlation between the selected column and 'Test Score'
    if column in correlation_matrix.columns:
        correlation_value = df_filtered['Test Score'].corr(df_filtered[column])
        print(f"Correlation between '{column}' and 'Test Score' for '{chapter}': {correlation_value}")

# Set up event listeners
upload_widget.observe(handle_upload, names='value')
chapter_dropdown.observe(display_correlation, names='value')
column_dropdown.observe(display_correlation, names='value')

# Display widgets
display(upload_widget, chapter_dropdown, column_dropdown)
