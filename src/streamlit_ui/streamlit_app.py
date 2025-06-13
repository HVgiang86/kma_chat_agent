import asyncio
import os
import sys

import streamlit as st
from langchain_core.messages import AIMessage

# Add the parent directory to sys.path to import our agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the agent
from agent.supervisor_agent import ReActGraph

st.set_page_config(page_title="KMA Agent Demo", page_icon="🤖", layout="wide", )

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Add debug toggle to session state
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False

# App title and description
st.title("KMA Agent Demo")

st.markdown("""
This is a demo of the KMA Agent, a langgraph-based agent that can help students with their questions about KMA regulations, their information, and their scores.
""")

# Chat interface
st.divider()
st.subheader("Chat with the KMA Agent")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Initialize chat rag in session state if it doesn't exist
if "chat_agent" not in st.session_state:
    with st.spinner("Đang khởi tạo trợ lý ảo..."):
        # Initialize chat rag
        st.session_state.chat_agent = ReActGraph()
        st.session_state.chat_agent.create_graph()
        st.session_state.chat_agent.print_mermaid()
        st.success("Trợ lý ảo đã sẵn sàng!")

# Get user input
user_query = st.chat_input("Type your question here...")

# Process user input
if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.write(user_query)

    # Show thinking spinner
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the agent
            try:
                response = asyncio.run(st.session_state.chat_agent.chat(user_query))
                last_res = response[-1]

                if isinstance(last_res, AIMessage):
                    messages = last_res.content

                # Check if we got a fallback response
                if messages == "I couldn't generate a response. Please try a different query.":
                    st.error(
                        "The agent couldn't generate a proper response. Please try a different query or check the agent's implementation.")

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": messages})
                st.write(messages)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with additional information
with st.sidebar:
    st.header("About this Demo")
    st.markdown("""
    This demo showcases a langgraph agent built for KMA students.

    """)

    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Add a debug mode toggle
    st.divider()
    st.subheader("Debug Options")
    debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
    if debug_mode != st.session_state.debug_mode:
        st.session_state.debug_mode = debug_mode
        st.rerun()
