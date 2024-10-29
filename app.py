import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Sidebar to choose the task: Summarization
st.sidebar.image("D:\\Fine Tunning\\nlp\\logo.png", use_column_width=True)

st.sidebar.header("Choose Task")
task = st.sidebar.selectbox(
    "Select a task",
    ["Summarization"]
)

# Function to load the models
def load_model(model_name):
    try:
        if model_name == "BART":
            model_path = 'D:\\Fine Tunning\\nlp\\bert\\model'
            tokenizer_path = 'D:\\Fine Tunning\\nlp\\bert\\tokenizer'
        elif model_name == "T5":
            model_path = 'D:\\Fine Tunning\\nlp\\T5_small\\model'
            tokenizer_path = 'D:\\Fine Tunning\\nlp\\T5_small\\tokenizer'
        elif model_name == "PEGASUS":
            model_path = 'D:\\Fine Tunning\\nlp\\pegasus\\Model'  # Correct path
            tokenizer_path = 'D:\\Fine Tunning\\nlp\\pegasus\\Tokenizer'  # Correct path

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        return model, tokenizer
    except OSError as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None

# Function to summarize text
def summarize_text(text, model, tokenizer):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Summarization functionality
if task == "Summarization":
    st.title("✨ NLP Text Summarization App")

    # Sidebar to select model
    model_name = st.sidebar.selectbox(
        "Select a model",
        ["BART", "T5", "PEGASUS"]
    )

    # Text input from the user
    text_input = st.text_area("Enter text for summarization", height=200)

    # Load the selected model and tokenizer
    model, tokenizer = load_model(model_name)

    # Summarize the text when the button is clicked
    if st.button("Summarize Text"):
        if text_input and model is not None:
            summary = summarize_text(text_input, model, tokenizer)
            st.markdown("### Summarization Output")
            st.markdown(f"""
                <div style="border-radius: 10px; padding: 15px; background-color: #eef7ff;">
                {summary}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please enter some text for summarization.")
