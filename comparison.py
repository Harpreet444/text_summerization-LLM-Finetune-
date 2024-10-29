import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import f1_score, precision_score, recall_score

# Function to load the models
def load_model(model_name):
    try:
        if model_name == "BERT":
            model_path = 'D:\\Fine Tunning\\nlp\\bert\\model'
            tokenizer_path = 'D:\\Fine Tunning\\nlp\\bert\\tokenizer'
        elif model_name == "T5":
            model_path = 'D:\\Fine Tunning\\nlp\\T5_small\\model'
            tokenizer_path = 'D:\\Fine Tunning\\nlp\\T5_small\\tokenizer'
        elif model_name == "PEGASUS":
            model_path = 'D:\\Fine Tunning\\nlp\\pegasus\\Model'
            tokenizer_path = 'D:\\Fine Tunning\\nlp\\pegasus\\Tokenizer'

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)  # Slow tokenizer to avoid warning

        return model, tokenizer
    except OSError as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

# Function to summarize text
def summarize_text(text, model, tokenizer):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Function to compare models on a dataset and compute metrics
def compare_models(data, bert_model, bert_tokenizer, t5_model, t5_tokenizer, pegasus_model, pegasus_tokenizer):
    data = data.head(10)  # Take the first 10 rows
    summaries = {"BERT": [], "T5": [], "PEGASUS": []}
    references = data['summary'].tolist()
    inputs = data['dialogue'].tolist()

    for text in inputs:
        # Get BERT summary
        if bert_model is not None:
            bert_summary = summarize_text(text, bert_model, bert_tokenizer)
            summaries["BERT"].append(bert_summary)
        else:
            summaries["BERT"].append("Error")

        # Get T5 summary
        if t5_model is not None:
            t5_summary = summarize_text(text, t5_model, t5_tokenizer)
            summaries["T5"].append(t5_summary)
        else:
            summaries["T5"].append("Error")

        # Get PEGASUS summary
        if pegasus_model is not None:
            pegasus_summary = summarize_text(text, pegasus_model, pegasus_tokenizer)
            summaries["PEGASUS"].append(pegasus_summary)
        else:
            summaries["PEGASUS"].append("Error")

    # Compute rough metrics (average F1, Precision, and Recall)
    metrics = {}
    for model_name in summaries:
        if "Error" not in summaries[model_name]:
            metrics[model_name] = {
                'F1 Score': f1_score(references, summaries[model_name], average='macro', zero_division=1),
                'Precision': precision_score(references, summaries[model_name], average='macro', zero_division=1),
                'Recall': recall_score(references, summaries[model_name], average='macro', zero_division=1)
            }
        else:
            metrics[model_name] = {
                'F1 Score': None,
                'Precision': None,
                'Recall': None
            }

    return metrics

# Function to plot the metrics in a bar chart using matplotlib
def plot_metrics(metrics):
    # Prepare data for plotting
    models = list(metrics.keys())
    f1_scores = [metrics[model]['F1 Score'] for model in models]
    precision_scores = [metrics[model]['Precision'] for model in models]
    recall_scores = [metrics[model]['Recall'] for model in models]

    bar_width = 0.2
    index = range(len(models))

    # Plot bars for F1 score, Precision, and Recall
    plt.bar(index, f1_scores, bar_width, label='F1 Score')
    plt.bar([i + bar_width for i in index], precision_scores, bar_width, label='Precision')
    plt.bar([i + 2 * bar_width for i in index], recall_scores, bar_width, label='Recall')

    # Adding labels and title
    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Model Comparison')
    plt.xticks([i + bar_width for i in index], models)
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.show()

# Main function to run the comparison
def main():
    # Load CSV data (replace with your own file path)
    data = pd.read_csv('D:\\Fine Tunning\\nlp\\test.csv').drop(['id'], axis='columns')

    # Ensure the CSV has the necessary columns
    if 'dialogue' in data.columns and 'summary' in data.columns:
        print("File loaded successfully. Starting comparison...")

        # Load all models and their respective tokenizers
        bert_model, bert_tokenizer = load_model("BERT")
        t5_model, t5_tokenizer = load_model("T5")
        pegasus_model, pegasus_tokenizer = load_model("PEGASUS")

        # Compare models on the dataset
        metrics = compare_models(data, bert_model, bert_tokenizer, t5_model, t5_tokenizer, pegasus_model, pegasus_tokenizer)

        # Print and plot metrics
        print("Model Comparison Metrics:")
        print(metrics)

        plot_metrics(metrics)
    else:
        print("CSV file must contain 'dialogue' and 'summary' columns.")

if __name__ == "__main__":
    main()
