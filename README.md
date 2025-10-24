# PragmatiCQA Assignment - Step-by-Step Setup Guide

This guide will walk you through setting up and running the complete PragmatiCQA assignment implementation.

## Prerequisites

Before starting, ensure you have:
- Python 3.11 or higher
- Git installed
- Access to xAI API (for Grok model)
- At least 8GB RAM (for model loading)
- Sufficient disk space for datasets (~1GB)

## Step 1: Environment Setup

### 1.1 Clone the Repository
```bash
# Create the main homework directory
mkdir hw3
cd hw3

# Clone the assignment repository
git clone https://github.com/melhadad/nlp-with-llms-2025-hw3.git

# Clone the PragmatiCQA dataset repository
git clone https://github.com/qipeng/PragmatiCQA.git
```

### 1.2 Download Dataset Sources
1. Download the PragmatiCQA sources from: https://drive.google.com/file/d/17vbeArdufh8rfhkg2I4Mwm0C9Og5klFR/view?usp=drive_link
2. Extract the downloaded file to create `PragmatiCQA-sources` directory
3. Place it in the `hw3` directory alongside the other folders

### 1.3 Set Up API Key
Create a `.env` file in the `nlp-with-llms-2025-hw3` directory:
```bash
cd nlp-with-llms-2025-hw3
echo "XAI_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual xAI API key.

## Step 2: Install Dependencies

### 2.1 Install UV (if not already installed)
```bash
# On Windows (PowerShell)
pip install uv

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2.2 Install Project Dependencies
```bash
cd nlp-with-llms-2025-hw3
uv sync
```

This will install all required packages including:
- dspy-ai
- sentence-transformers
- transformers
- beautifulsoup4
- faiss-cpu
- And other dependencies

## Step 3: Verify Directory Structure

Your directory structure should look like this:
```
hw3/
├── nlp-with-llms-2025-hw3/
│   ├── .env
│   ├── pragmaticqa_complete.ipynb
│   ├── pyproject.toml
│   └── ... (other files)
├── PragmatiCQA/
│   └── data/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
└── PragmatiCQA-sources/
    ├── The Legend of Zelda/
    ├── Batman/
    ├── Doctor Who/
    └── ... (other topic folders)
```

## Step 4: Run the Complete Implementation

### 4.1 Launch Jupyter Notebook
```bash
cd nlp-with-llms-2025-hw3
uv run jupyter notebook
```

### 4.2 Open the Main Notebook
1. Open `pragmaticqa_complete.ipynb`
2. Make sure the kernel is set to `nlp-with-llms-2025-hw3`

### 4.3 Execute the Notebook
Run all cells in order. The notebook is organized as follows:

#### Part 0: Dataset Analysis
- Loads and analyzes the PragmatiCQA dataset
- Demonstrates pragmatic phenomena with sample conversations
- Provides theoretical background

#### Part 1: Traditional NLP Approach
- Sets up DistilBERT QA model
- Implements three configurations:
  - **Literal**: Uses only literal spans from dataset
  - **Pragmatic**: Uses pragmatic spans from dataset
  - **Retrieved**: Uses retrieved context from HTML sources
- Evaluates using SemanticF1 metric

#### Part 2: LLM Multi-Step Prompting
- Implements sophisticated DSPy modules:
  - `ConversationAnalyzer`: Analyzes conversation history
  - `PragmaticReasoner`: Reasons about additional information needs
  - `CooperativeAnswerGenerator`: Generates cooperative answers
  - `PragmaticRAG`: Main orchestrating module
- Evaluates on both first questions and full conversations
- Compares performance with traditional approach

## Step 5: Understanding the Results

### 5.1 Expected Outputs
The notebook will generate:
- Dataset analysis with sample conversations
- Performance metrics for traditional QA (3 configurations)
- Performance metrics for pragmatic RAG
- Comparison tables
- Cost analysis
- Discussion of results

### 5.2 Key Metrics
- **SemanticF1 Score**: Measures semantic similarity between predicted and reference answers
- **Configuration Comparison**: Shows how different context sources affect performance
- **Turn-by-Turn Analysis**: Shows how conversational context improves later questions

## Step 6: Troubleshooting

### 6.1 Common Issues

**Issue**: "ModuleNotFoundError" for dspy or other packages
**Solution**: 
```bash
uv sync --reinstall
```

**Issue**: "XAI_API_KEY not found"
**Solution**: 
- Check that `.env` file exists in `nlp-with-llms-2025-hw3/`
- Verify API key is correct
- Restart Jupyter kernel

**Issue**: "No documents found for topic"
**Solution**: 
- Verify `PragmatiCQA-sources` directory exists
- Check that topic folders contain HTML files
- Ensure proper directory structure

**Issue**: Out of memory errors
**Solution**: 
- Reduce batch sizes in evaluation
- Use smaller sample sizes
- Close other applications

### 6.2 Performance Optimization

**For Faster Execution:**
- Reduce the number of topics processed (modify `create_topic_retrievers`)
- Use smaller sample sizes for evaluation
- Process only first questions initially

**For Better Results:**
- Increase the number of retrieved passages (k parameter)
- Use more sophisticated retrieval models
- Implement better chunking strategies

## Step 7: Customization and Extensions

### 7.1 Experiment with Different Models
- Try different LLM providers (OpenAI, Anthropic, etc.)
- Experiment with different embedding models
- Test different retrieval strategies

### 7.2 Modify Evaluation
- Add more evaluation metrics
- Implement custom evaluation functions
- Analyze specific conversation patterns

### 7.3 Extend the Framework
- Add more sophisticated pragmatic reasoning
- Implement follow-up question prediction
- Create interactive conversation interfaces

## Step 8: Submission

### 8.1 Prepare Your Submission
1. Ensure all cells in `pragmaticqa_complete.ipynb` have been executed
2. Save the notebook with outputs
3. Review the discussion questions and add your insights

### 8.2 Key Components to Include
- Dataset analysis and sample conversations
- Complete implementation of both approaches
- Evaluation results and comparisons
- Answers to discussion questions
- Analysis of Theory of Mind capabilities

## Additional Resources

### Documentation
- [DSPy Documentation](https://dspy.ai/)
- [PragmatiCQA Paper](https://github.com/qipeng/PragmatiCQA)
- [SemanticF1 Evaluation](https://dspy.ai/api/evaluation/SemanticF1/)

### Related Papers
- PragmatiCQA: A Dataset for Pragmatic Question Answering
- DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines
- Theory of Mind in Large Language Models

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are properly installed
4. Verify your directory structure matches the expected layout

Good luck with your assignment! 🚀
