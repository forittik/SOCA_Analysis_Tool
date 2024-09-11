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
        background-color: #f0f2f6;
        color: #000000;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        color: #000000;
    }
    h1, h2, h3, h4 {
        color: #333333;
        font-family: 'Segoe UI';
    }
    .stButton>button {
        background-color: #007bff; color: white; border-radius: 5px;
    }
    .stSelectbox>div>div>div {
        font-family: 'Segoe UI'; 
        font-size: 14px;
        color: #333333;
    }
    .stSelectbox>div>div>div[data-baseweb="select"]>div {
        color: #333333;
    }
    .stMarkdown {
        color: #000000;
    }
    .stPlot {
        background-color: #ffffff;
    }
    .streamlit-expander
