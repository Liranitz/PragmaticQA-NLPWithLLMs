Memory Bank: dspy.evaluate.SemanticF1

Overview:
----------
`dspy.evaluate.SemanticF1` is a metric used to evaluate the semantic similarity between predicted and reference text outputs. It is particularly useful in natural language generation tasks where exact string matching is too strict and semantic equivalence is more appropriate.

Purpose:
---------
To compute a semantic F1 score that reflects the overlap in meaning between a predicted output and a reference output, rather than relying solely on lexical similarity.

Implementation Details:
------------------------
- The metric uses embeddings (e.g., from a language model) to represent text.
- It computes precision and recall based on the semantic similarity of tokens or phrases.
- The F1 score is then derived from these precision and recall values.

Mathematical Formulation:
--------------------------
Let:
- P = set of embeddings for predicted tokens
- R = set of embeddings for reference tokens

Then:
- Precision = average max similarity of each token in P to any token in R
- Recall = average max similarity of each token in R to any token in P
- F1 = 2 * (Precision * Recall) / (Precision + Recall)

Use Cases:
----------
- Evaluating chatbot responses
- Summarization quality assessment
- Paraphrase generation evaluation
- Any NLP task where semantic fidelity is more important than exact match

Limitations:
------------
- Requires a good embedding model to capture semantic similarity
- May be computationally expensive for long texts
- Sensitive to the quality of tokenization and embedding granularity

Example Code:
-------------
from dspy.evaluate import SemanticF1

metric = SemanticF1()
score = metric(prediction="The cat sat on the mat", reference="A cat is sitting on a mat")
print("Semantic F1 Score:", score)

Command-Line Tips:
------------------
- Store this file as `semanticf1_memory_bank.txt`
- Use `cat semanticf1_memory_bank.txt` or `less semanticf1_memory_bank.txt` to view it
- Keep it in a known directory and symlink it to your CLI tools if needed

