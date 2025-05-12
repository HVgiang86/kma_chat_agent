#!/usr/bin/env python3
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from retriever import create_hybrid_retriever
from graph import KMAChatAgent

# Load environment variables
load_dotenv()

# Define paths
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent.parent
vector_db_path = os.path.join(project_root, "vector_db")
data_path = os.path.join(project_root, "data", "regulation.txt")

# Page config
st.set_page_config(
    page_title="KMA Regulations Assistant",
    page_icon="🎓",
    layout="centered"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .user-bubble {
        background-color: #E3F2FD;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 80%;
    }
    .assistant-bubble {
        background-color: #F5F5F5;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 10px;
        display: inline-block;
        max-width: 80%;
        float: right;
    }
    .clear-button {
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>KMA Regulations Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Hỏi đáp về quy định của Học viện Kỹ thuật Mật mã</p>", unsafe_allow_html=True)

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize chat agent in session state if it doesn't exist
if "chat_agent" not in st.session_state:
    with st.spinner("Đang khởi tạo trợ lý ảo..."):
        # Create hybrid retriever
        hybrid_retriever, _ = create_hybrid_retriever(
            vector_db_path=vector_db_path,
            data_path=data_path
        )
        
        # Initialize chat agent
        st.session_state.chat_agent = KMAChatAgent(hybrid_retriever)
        st.success("Trợ lý ảo đã sẵn sàng!")

# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='assistant-bubble'>{message['content']}</div>", unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Nhập câu hỏi của bạn...")

# Process the input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    st.markdown(f"<div class='user-bubble'>{user_input}</div>", unsafe_allow_html=True)
    
    # Get response from the assistant
    with st.spinner("Đang xử lý..."):
        response = st.session_state.chat_agent.chat(user_input)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response
    st.markdown(f"<div class='assistant-bubble'>{response}</div>", unsafe_allow_html=True)

# Clear chat button
if st.session_state.messages:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Xóa cuộc trò chuyện", key="clear"):
            st.session_state.messages = []
            st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("Developed by KMA AI Lab") 