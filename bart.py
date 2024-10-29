import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

# Load the fine-tuned BART model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("D:\\Fine Tunning\\nlp\\pegasus\\Model")
tokenizer = AutoTokenizer.from_pretrained("D:\\Fine Tunning\\nlp\\pegasus\\Tokenizer")

# Load your test dataset
test_data = pd.read_csv('D:\\Fine Tunning\\nlp\\samsum-test.csv')

# Evaluate on a random subset of 50 examples (adjust this number if needed)
test_sample_size = 50
test_data_sample = test_data.sample(test_sample_size)

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize variables to store total scores
total_rouge1 = 0
total_rouge2 = 0
total_rougeL = 0

# Lists to store ground truth and predicted summaries for evaluation
y_true = []
y_pred = []

# Loop through the sampled test data with a counter for correct indexing
for count, (_, row) in enumerate(test_data_sample.iterrows(), 1):
    # Input text (document to be summarized)
    text = row['dialogue']  # Assuming the text is in a column named 'dialogue'
    reference_summary = row['summary']  # Assuming reference summary is in a column named 'summary'

    # Tokenize the input text with reduced max_length
    inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)

    # Generate summary using BART with fewer beams for speed
    summary_ids = model.generate(inputs["input_ids"], max_length=100, num_beams=2, early_stopping=True)

    # Decode the generated summary
    generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Calculate ROUGE scores between generated and reference summary
    scores = scorer.score(reference_summary, generated_summary)

    # Accumulate ROUGE scores
    total_rouge1 += scores['rouge1'].fmeasure
    total_rouge2 += scores['rouge2'].fmeasure
    total_rougeL += scores['rougeL'].fmeasure

    # Store ground truth and predictions for evaluation
    y_true.append(reference_summary)
    y_pred.append(generated_summary)

    # Print sequential progress
    print(f"Processed {count}/{test_sample_size} examples.")

# Calculate average ROUGE scores
average_rouge1 = total_rouge1 / test_sample_size
average_rouge2 = total_rouge2 / test_sample_size
average_rougeL = total_rougeL / test_sample_size

# Convert summaries to a format suitable for precision/recall/F1 calculations
# Tokenization
def tokenize_summaries(summaries):
    return [set(tokenizer.tokenize(summary)) for summary in summaries]

y_true_tokens = tokenize_summaries(y_true)
y_pred_tokens = tokenize_summaries(y_pred)

# Calculate precision, recall, and F1 using sets for token overlap
def calculate_metrics(y_true_tokens, y_pred_tokens):
    precision_list = []
    recall_list = []
    f1_list = []

    for true_set, pred_set in zip(y_true_tokens, y_pred_tokens):
        if len(pred_set) == 0:  # No predicted tokens
            precision_list.append(0)
            recall_list.append(1 if len(true_set) > 0 else 0)
            f1_list.append(0)
            continue

        true_positive = len(true_set.intersection(pred_set))
        precision = true_positive / len(pred_set)
        recall = true_positive / len(true_set)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)

# Calculate the metrics
precision, recall, f1 = calculate_metrics(y_true_tokens, y_pred_tokens)

# Print the average ROUGE scores and other metrics
print("\nAverage ROUGE Scores on Sampled Test Data:")
print(f"ROUGE-1: {average_rouge1:.4f}")
print(f"ROUGE-2: {average_rouge2:.4f}")
print(f"ROUGE-L: {average_rougeL:.4f}")
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Prepare data for bar chart
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
scores = [accuracy_score(y_true, y_pred), precision, recall, f1, average_rouge1, average_rouge2, average_rougeL]

# Create a bar chart for the scores
# Create a bar chart for the scores
plt.figure(figsize=(10, 5))
plt.bar(metrics, scores, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan'])
plt.ylabel('Score')
plt.title('Pegasus Evaluation Metrics')  # Updated title
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.grid(axis='y')
plt.show()


plt.figure(figsize=(10, 5))
plt.bar(metrics, scores, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'cyan'])
plt.ylabel('Score')
plt.title('Evaluation Metrics')
plt.xticks(rotation=45)
plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
plt.grid(axis='y')
plt.show()
