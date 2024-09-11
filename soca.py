import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Set page config with a custom theme
st.set_page_config(page_title="📊 Advanced Data Analysis App", layout="wide")

# Custom CSS for advanced styling with improved text visibility
st.markdown("""
    <style>
    .main {
        background-color: #f11000;
        color: #000000;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        color: #000000;
    }
    h1, h2, h3, h4 {
        color: #000000;
        font-family: 'Segoe UI';
    }
    .stButton>button {
        background-color: #007bff; color: #000000; border-radius: 5px;
    }
    .stSelectbox>div>div>div {
        font-family: 'Segoe UI'; 
        font-size: 14px;
        color: #000000;
    }
    .stSelectbox>div>div>div[data-baseweb="select"]>div {
        color: #000000;
    }
    .stSelectbox>div>div>div[data-baseweb="select"]>div[data-option="Optics"],
    .stSelectbox>div>div>div[data-baseweb="select"]>div[data-option="Strength"] {
        color: #f11000;
    }
    .stMarkdown {
        color: #000000;
    }
    .stPlot {
        background-color: #ffffff;
    }
    .streamlit-expanderHeader {
        color: #000000;
    }
    label.css-mkogse.e16fv1kl2 {
        color: #000000;
    }
    .stTextInput>div>div>input {
        color: #000000;
    }
    .stTextInput>label {
        color: #000000;
    }
    .stRadio>div {
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Upload Data", "📈 Performance Analysis", "🔍 Skills Analysis", "🔗 Correlation Analysis", "📚 Chapter Statistics"])

# File upload
def handle_upload():
    st.markdown("### 📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['Test Score'] = pd.to_numeric(df['Test Score'], errors='coerce')
        st.session_state.df = df
        st.success("✅ Data loaded successfully!")
        st.write(df.head())

# Performance Analysis
def performance_analysis():
    if st.session_state.df.empty:
        st.warning("⚠ Please upload data first.")
        return

    chapter = st.selectbox("Select Test Chapter", options=st.session_state.df['Test Chapter'].unique())
    
    avg_score_by_chapter = st.session_state.df.groupby('Test Chapter')['Test Score'].mean().reset_index()
    
    st.markdown(f"## 📈 Performance Analysis for '{chapter}'")
    st.write(avg_score_by_chapter[avg_score_by_chapter['Test Chapter'] == chapter])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Test Chapter', y='Test Score', data=avg_score_by_chapter, palette='viridis', ax=ax)
    plt.title('📊 Average Test Score by Chapter', fontsize=16, color='black')
    plt.xlabel('Test Chapter', fontsize=14, color='black')
    plt.ylabel('Average Test Score', fontsize=14, color='black')
    plt.xticks(rotation=45, color='black')
    plt.yticks(color='black')
    st.pyplot(fig)

    # Score Distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    test_scores = st.session_state.df[st.session_state.df['Test Chapter'] == chapter]['Test Score'].dropna().values
    plt.hist(test_scores, bins=15, edgecolor='black', color='#007bff')
    plt.title(f'📉 Distribution of Test Scores for {chapter}', fontsize=16, color='black')
    plt.xlabel('Test Score', fontsize=14, color='black')
    plt.ylabel('Frequency', fontsize=14, color='black')
    st.pyplot(fig)

# Skills Analysis
def skills_analysis():
    if st.session_state.df.empty:
        st.warning("⚠ Please upload data first.")
        return

    chapter = st.selectbox("Select Test Chapter", options=st.session_state.df['Test Chapter'].unique())
    
    df = st.session_state.df[st.session_state.df['Test Chapter'] == chapter].copy()
    
    # Skill Frequency Analysis
    skill_frequency = pd.concat([df['Strength'], df['Opportunity'], df['Challenge']]).value_counts().reset_index()
    skill_frequency.columns = ['Skill', 'Frequency']
    
    st.markdown(f"## 🔍 Skills Analysis for '{chapter}'")
    st.write(skill_frequency)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x='Frequency', y='Skill', data=skill_frequency, palette='Set2', ax=ax)
    plt.title(f'🧠 Frequency of Skills for {chapter}', fontsize=16, color='black')
    plt.xlabel('Frequency', fontsize=14, color='black')
    plt.ylabel('Skill', fontsize=14, color='black')
    st.pyplot(fig)

# Correlation Analysis
def correlation_analysis():
    if st.session_state.df.empty:
        st.warning("⚠ Please upload data first.")
        return

    chapter = st.selectbox("Select Test Chapter", options=st.session_state.df['Test Chapter'].unique())
    column = st.selectbox("Correlate with", options=['Strength', 'Opportunity', 'Challenge'])
    
    df_filtered = st.session_state.df[st.session_state.df['Test Chapter'] == chapter].copy()
    
    label_encoder = LabelEncoder()
    df_filtered['Strength_encoded'] = label_encoder.fit_transform(df_filtered['Strength'].astype(str))
    df_filtered['Opportunity_encoded'] = label_encoder.fit_transform(df_filtered['Opportunity'].astype(str))
    df_filtered['Challenge_encoded'] = label_encoder.fit_transform(df_filtered['Challenge'].astype(str))

    correlation_matrix = df_filtered[['Test Score', 'Strength_encoded', 'Opportunity_encoded', 'Challenge_encoded']].corr()

    st.markdown(f"## 🔗 Correlation Analysis for '{chapter}'")
    st.write(correlation_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, cbar=True, square=True, fmt=".2f", ax=ax)
    plt.title(f"🔥 Correlation Heatmap for '{chapter}'", fontsize=16, color='black')
    st.pyplot(fig)
    
    column_encoded = f"{column}_encoded"
    if column_encoded in correlation_matrix.columns:
        correlation_value = df_filtered['Test Score'].corr(df_filtered[column_encoded])
        st.write(f"Correlation between '{column}' and 'Test Score' for '{chapter}': {correlation_value:.2f}")

# Chapter Statistics
def chapter_statistics():
    if st.session_state.df.empty:
        st.warning("⚠ Please upload data first.")
        return

    chapter = st.selectbox("Select Test Chapter", options=st.session_state.df['Test Chapter'].unique())
    
    filtered_df = st.session_state.df[st.session_state.df['Test Chapter'] == chapter]
    avg_score = filtered_df['Test Score'].mean()
    num_entries = len(filtered_df)
    
    st.markdown(f"## 📚 Chapter Statistics for '{chapter}'")
    st.write(f"Average Test Score: {avg_score:.2f}")
    st.write(f"Number of Entries: {num_entries}")
    
    # Plot score distribution by Strength, Opportunity, and Challenge
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    strength_scores = filtered_df.groupby('Strength')['Test Score'].mean()
    strength_scores.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('💪 Score Distribution by Strength', fontsize=14, color='black')
    ax1.set_xlabel('Strength', fontsize=12, color='black')
    ax1.set_ylabel('Average Test Score', fontsize=12, color='black')
    ax1.tick_params(axis='x', rotation=45)
    
    opportunity_scores = filtered_df.groupby('Opportunity')['Test Score'].mean()
    opportunity_scores.plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title('🚀 Score Distribution by Opportunity', fontsize=14, color='black')
    ax2.set_xlabel('Opportunity', fontsize=12, color='black')
    ax2.set_ylabel('Average Test Score', fontsize=12, color='black')
    ax2.tick_params(axis='x', rotation=45)
    
    challenge_scores = filtered_df.groupby('Challenge')['Test Score'].mean()
    challenge_scores.plot(kind='bar', ax=ax3, color='lightcoral')
    ax3.set_title('🏋 Score Distribution by Challenge', fontsize=14, color='black')
    ax3.set_xlabel('Challenge', fontsize=12, color='black')
    ax3.set_ylabel('Average Test Score', fontsize=12, color='black')
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

# Main app logic
def main():
    if page == "🏠 Upload Data":
        st.title("🏠 Upload Data")
        handle_upload()
    elif page == "📈 Performance Analysis":
        st.title("📈 Performance Analysis")
        performance_analysis()
    elif page == "🔍 Skills Analysis":
        st.title("🔍 Skills Analysis")
        skills_analysis()
    elif page == "🔗 Correlation Analysis":
        st.title("🔗 Correlation Analysis")
        correlation_analysis()
    elif page == "📚 Chapter Statistics":
        st.title("📚 Chapter Statistics")
        chapter_statistics()
    
if __name__ == "__main__":
    main()
