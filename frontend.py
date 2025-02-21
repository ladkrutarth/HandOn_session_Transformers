import streamlit as st
import requests

# Backend API URL
BACKEND_URL = "http://127.0.0.1:5001"

st.title("BERT Question Answering and Attention Visualization")

# **Question Answering Section**
st.header("Question Answering with BERT")
context = st.text_area("Enter context:", "Big data analytics helps in finding patterns in data.")
question = st.text_input("Enter your question:", "What does big data analytics do?")

if st.button("Get Answer"):
    if context and question:
        response = requests.post(f"{BACKEND_URL}/qa", json={"question": question, "context": context})
        if response.status_code == 200:
            data = response.json()
            st.success(f"**Answer:** {data['answer']}")
            st.write(f"**Confidence Score:** {data['confidence']}%")
        else:
            st.error("Error processing request")
    else:
        st.warning("Please enter both context and question.")

# **Attention Visualization Section**
st.header("BERT Attention Visualization")
sample_text = st.text_area("Enter text for visualization:", "I am studying Machine Learning.")

if st.button("Visualize Attention"):
    if sample_text:
        response = requests.post(f"{BACKEND_URL}/attention", json={"text": sample_text})
        if response.status_code == 200:
            data = response.json()
            st.success(f"**Attention Visualization Ready!**")
            st.write(f"**Number of Layers:** {data['num_layers']}")
            st.write(f"**Attention Shape:** {data['attention_shape']}")
        else:
            st.error("Error processing request")
    else:
        st.warning("Please enter text to visualize.")

