import streamlit as st
import home, task 

# Page config
st.set_page_config(page_title="YOLOv8 Object Detection", page_icon="🔎", layout="wide", menu_items=None)

# Sidebar config 
page_names_to_funcs = {
    "Home": home.home,
    "Detection": task.task
}

# Navigate pages 
demo_name = st.sidebar.radio("Select a page:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()