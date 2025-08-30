
DSPy Tutorial: Multi-Hop Retrieval
==================================

Overview:
---------
This tutorial demonstrates how to build a DSPy module for multi-hop retrieval using a combination of sub-modules and a retrieval corpus. The goal is to construct a system that can answer complex claims by retrieving relevant Wikipedia pages.

Key Steps:
----------
1. **Installation and Setup**
   - Install DSPy and required libraries: `pip install -U dspy bm25s PyStemmer "jax[cpu]"`
   - Download and decompress the Wikipedia abstracts corpus (5 million pages, ~500MB).

2. **Model Configuration**
   - Use Meta's Llama-3.1-8B-Instruct as the main LM.
   - Optionally configure GPT-4o as a teacher model for optimization.

3. **Corpus Preparation**
   - Load and tokenize the Wikipedia abstracts using BM25S and PyStemmer.
   - Index the corpus for retrieval.

4. **Dataset Usage**
   - Load the HoVer dataset from Hugging Face.
   - Filter examples with 3-hop claims and prepare train/dev/test splits.

5. **Module Design**
   - Define a `Hop` module with two sub-modules:
     - `generate_query`: Generates a search query from claim and notes.
     - `append_notes`: Updates notes and extracts titles from retrieved context.
   - The module iteratively refines notes and titles over multiple hops.

6. **Evaluation**
   - Define `top5_recall` metric to measure retrieval accuracy.
   - Evaluate the initial program using DSPy's `Evaluate`.

7. **Optimization**
   - Use `MIPROv2` optimizer with GPT-4o to improve prompt performance.
   - Achieve significant improvement in recall (from ~30% to ~60%).

8. **Saving and Loading**
   - Save the optimized program to a JSON file.
   - Load and reuse the program for future inference.
-- 
Here’s the full content of the **Tutorial: Multi-Hop Retrieval** section from the DSPy documentation, formatted for easy copying and pasting into your memory bank file:

---

**Tutorial: Multi-Hop Retrieval**

Let’s walk through a quick example of building a `dspy.Module` with multiple sub-modules. We’ll do this for the task of multi-hop search.

---

### Step 1: Installation

Install the latest DSPy and datasets:

```bash
pip install -U dspy
pip install datasets
```

Optional: Set up MLflow Tracing to understand what’s happening under the hood.

---

### Step 2: Model Setup

We’ll use a small local LM, Meta’s Llama-3.1-8B-Instruct. You can host it via:

- Ollama (local laptop)
- SGLang (GPU server)
- Providers like Databricks or Together

We’ll also set up a larger LM (e.g., GPT-4o) as a teacher to help guide the small LM.

```python
import dspy

# Configure small LM
small_lm = dspy.Ollama(model='llama3:8b-instruct')

# Configure large LM as teacher
teacher_lm = dspy.OpenAI(model='gpt-4o')

# Set the small LM as default
dspy.settings.configure(lm=small_lm)
```

---

### Step 3: Define Submodules

We’ll define two submodules: one for retrieving documents and one for answering questions.

```python
class Retriever(dspy.Signature):
    """Retrieve documents relevant to a question."""
    question = dspy.Input()
    documents = dspy.Output()

class Answerer(dspy.Signature):
    """Answer a question using retrieved documents."""
    question = dspy.Input()
    documents = dspy.Input()
    answer = dspy.Output()
```

---

### Step 4: Build Multi-Hop Module

```python
class MultiHopQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.retrieve = dspy.Predict(Retriever)
        self.answer = dspy.Predict(Answerer)

    def forward(self, question):
        docs = self.retrieve(question=question).documents
        return self.answer(question=question, documents=docs)
```

---

### Step 5: Run and Evaluate

```python
qa = MultiHopQA()
response = qa.forward("Who is the CEO of the company that owns Instagram?")
print(response.answer)
```

---

Let me know if you’d like this converted to Markdown, PDF, or integrated into a specific format for your workflow.