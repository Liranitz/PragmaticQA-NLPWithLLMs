DSPy Memory Bank Summary

Overview:
DSPy is a framework for building, evaluating, and optimizing AI programs using language models (LMs). It provides modular components, evaluation tools, and optimization strategies to streamline the development of high-quality AI systems.

Key Concepts:
- DSPy Modules: Encapsulate LM behavior using structured input/output signatures.
- Signatures: Define the schema for inputs and outputs in DSPy modules.
- ChainOfThought, ProgramOfThought, ReAct: Built-in modules for reasoning and interaction.
- Retrieval-Augmented Generation (RAG): Combines retrieval with generation for improved responses.
- Optimizers: Tools like MIPROv2 to enhance prompts and system performance.

Tutorial Highlights:
- Basic RAG: Build a question-answering system using retrieved documents.
- Finetuning and Evaluation: Use metrics like Semantic F1 to assess system quality.
- Prompt Optimization: Improve system outputs through structured prompt tuning.
- Real-World Examples: Applications in financial analysis, code generation, and creative AI games.

Evaluation Strategies:
- Semantic F1: Measures overlap of key ideas between system and ground truth responses.
- DSPy Evaluate: Parallel evaluation with progress tracking and result tables.
- Inspect History: View structured LM interactions for debugging and analysis.

Optimization Techniques:
- MIPROv2: Learns from training/validation examples to optimize prompts.
- Prompt Compilation: Enhances module behavior through example-driven tuning.
- Cost Tracking: Monitor API usage and optimize for efficiency.

Use Cases:
- Tech QA: Answering technical questions using RAG and optimized prompts.
- Email Extraction: Parsing structured data from emails.
- Game Development: Building interactive text-based games.
- Code Generation: Generating code for unfamiliar libraries.

Deployment and Tools:
- MLflow Integration: Track experiments and evaluation results.
- Embedding-based Retrieval: Use OpenAI embeddings with FAISS or brute-force search.
- Saving/Loading: Serialize and reuse optimized DSPy programs.

Here is the full content of the **Tutorial: Retrieval-Augmented Generation (RAG)** section from the DSPy documentation, formatted for easy copying into your memory bank file:

---

### **Tutorial: Retrieval-Augmented Generation (RAG)**

Let’s walk through a quick example of basic question answering with and without retrieval-augmented generation (RAG) in DSPy. Specifically, let's build a system for answering Tech questions, e.g. about Linux or iPhone apps.

---

### **Step 1: Installation**

Install the latest DSPy and datasets:

```bash
pip install -U dspy datasets
```

---

### **Step 2: Configure DSPy Environment**

```python
import dspy
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

---

### **Step 3: Basic DSPy Module**

```python
qa = dspy.Predict('question: str -> response: str')
response = qa(question="what are high memory and low memory on linux?")
print(response.response)
```

---

### **Step 4: Chain of Thought Module**

```python
cot = dspy.ChainOfThought('question -> response')
cot(question="should curly braces appear on their own line?")
```

---

### **Step 5: Load Dataset for Evaluation**

```python
import ujson
from dspy.utils import download

download("https://huggingface.co/dspy/cache/resolve/main/ragqa_arena_tech_examples.jsonl")
with open("ragqa_arena_tech_examples.jsonl") as f:
    data = [ujson.loads(line) for line in f]
```

---

### **Step 6: Prepare Examples**

```python
data = [dspy.Example(**d).with_inputs('question') for d in data]
```

---

### **Step 7: Split Data**

```python
import random
random.Random(0).shuffle(data)
trainset, devset, testset = data[:200], data[200:500], data[500:1000]
```

---

### **Step 8: Semantic Evaluation**

```python
from dspy.evaluate import SemanticF1
metric = SemanticF1(decompositional=True)
pred = cot(**example.inputs())
score = metric(example, pred)
```

---

### **Step 9: Setup Retriever**

```python
max_characters = 6000
topk_docs_to_retrieve = 5
with open("ragqa_arena_tech_corpus.jsonl") as f:
    corpus = [ujson.loads(line)['text'][:max_characters] for line in f]

embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
search = dspy.retrievers.Embeddings(embedder=embedder, corpus=corpus, k=topk_docs_to_retrieve)
```

---

### **Step 10: Build RAG Module**

```python
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')
    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
```

---

### **Step 11: Evaluate RAG**

```python
rag = RAG()
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=24)
evaluate(rag)
```

---

### **Step 12: Optimize RAG**

```python
tp = dspy.MIPROv2(metric=metric, auto="medium", num_threads=24)
optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
```

---

### **Step 13: Save and Load Optimized Program**

```python
optimized_rag.save("optimized_rag.json")
loaded_rag = RAG()
loaded_rag.load("optimized_rag.json")
```

---

Let me know if you'd like this converted to Markdown or PDF, or if you want help integrating it into a specific workflow.