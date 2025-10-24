
# PragmatiCQA: Cooperative Question Answering with LLMs

This repository implements and benchmarks pragmatic, cooperative question answering using large language models (LLMs) and the DSPy framework. The project explores how conversational AI can go beyond literal answers to anticipate user needs, enrich responses, and demonstrate Theory of Mind-like reasoning.

## Overview

PragmatiCQA is a challenging benchmark for conversational QA, requiring models to:
- Answer questions using both literal and pragmatic context from a corpus.
- Anticipate follow-up questions and proactively provide relevant information.
- Reason about user intent and conversational history.

We compare two approaches:
1. **Traditional QA Baseline**: Uses a pre-trained extractive QA model (DistilBERT) with various context sources.
2. **LLM Multi-Step Reasoning**: Uses DSPy to build a multi-step pipeline leveraging LLMs for pragmatic, cooperative answers.

## Dataset

- **PragmatiCQA**: Open-domain conversational QA dataset (Fandom trivia, ~800 conversations).
- Each conversation includes literal and pragmatic answer spans, full dialog history, and human ratings.
- [Dataset & Paper](https://github.com/qipeng/PragmatiCQA)

## Approach

### Traditional QA Baseline
- Uses DistilBERT (SQuAD) to answer questions from:
  - Literal spans
  - Pragmatic spans
  - Retrieved context (from HTML sources)
- Evaluated with SemanticF1 (LLM-as-judge for answer quality)

### LLM Multi-Step Reasoning (DSPy)
- Custom DSPy modules:
  - `ConversationAnalyzer`: Summarizes user interests and goals
  - `PragmaticReasoner`: Infers pragmatic needs and generates cooperative queries
  - `CooperativeAnswerGenerator`: Synthesizes enriched answers
  - `PragmaticRAG`: Orchestrates multi-step retrieval and answer generation
- Evaluates both first-turn and multi-turn conversational answers

## Results

| Model / Context         | Semantic F1 |
|------------------------ |:-----------:|
| Baseline (Literal)      |    0.42     |
| Baseline (Pragmatic)    |    0.38     |
| Baseline (Retrieved)    |    0.15     |
| PragmaticRAG (First Q)  |    0.33     |
| PragmaticRAG (All Qs)   |    0.31     |

**Key Findings:**
- LLM-based multi-step reasoning substantially outperforms traditional QA in realistic settings.
- Cooperative answers are longer, more informative, and context-aware.
- The model demonstrates partial Theory of Mind by anticipating user needs and clarifying intent.

## Example

**Q:** "Which is the most dangerous dinosaur?"

- *Baseline (Literal):* "Velociraptor."
- *PragmaticRAG:* "Velociraptor is considered dangerous, as evidenced by attacks on other dinosaurs like Protoceratops. T-Rex, one of the largest carnivores, might have been the biggest terrestrial predator of all time."

## How It Works

See [`pragmaticqa_complete.ipynb`](pragmaticqa_complete.ipynb) for full code, experiments, and analysis.

## References

- [DSPy Documentation](https://dspy.ai/)
- [PragmatiCQA Paper & Dataset](https://github.com/qipeng/PragmatiCQA)
- [SemanticF1 Metric](https://dspy.ai/api/evaluation/SemanticF1/)

