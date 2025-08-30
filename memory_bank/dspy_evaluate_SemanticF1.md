## 🧠 Memory Bank Data for Cline

**Component Name**: `dspy.evaluate.SemanticF1`  
**Purpose**: Evaluate the semantic similarity between predicted and reference answers in DSPy.  
**Use Case**: Used in DSPy to assess the quality of generated answers, especially in question-answering tasks.  
**Key Concepts**:
- Semantic F1 score  
- Decompositional evaluation  
- Integration with DSPy modules  
- Evaluation of Chain-of-Thought and RAG modules  

---

## 📘 Tutorial: `dspy.evaluate.SemanticF1`

### Step 1: Import SemanticF1

```python
from dspy.evaluate import SemanticF1
```

### Step 2: Initialize the Metric

```python
metric = SemanticF1(decompositional=True)
```

- `decompositional=True` enables a more fine-grained evaluation by comparing subcomponents of the answer.

### Step 3: Evaluate a Prediction

```python
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
)

prediction = dspy.Prediction(
    answer="The capital of France is Paris."
)

score = metric(example, prediction)
print(score)
```

### Step 4: Use in Evaluation Pipeline

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(
    devset=devset,
    metric=metric,
    num_threads=24
)

evaluate(my_module)
```

This integrates `SemanticF1` into a full evaluation loop for a DSPy module like a `ChainOfThought` or `RAG` system.
