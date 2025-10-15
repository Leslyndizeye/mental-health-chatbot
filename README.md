# Mental Health Support Chatbot

An AI-powered conversational agent that provides emotional support using natural language processing. Built by fine-tuning GPT-2 on mental health counseling conversations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)
![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-green.svg)

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Technical Challenges](#technical-challenges)
- [Training Process](#training-process)
- [Results & Evaluation](#results--evaluation)
- [How to Run](#how-to-run)
- [Limitations](#limitations)
- [References](#references)

## Problem Statement

Mental health support faces critical barriers including limited therapist availability, high costs, social stigma, and lack of 24/7 support. This project develops an AI chatbot that provides accessible, immediate emotional support while recognizing mental health concerns.

**Note:** This is not a replacement for professional therapy.

## Dataset

**Source:** [Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) (Hugging Face)

**Composition:**
- Original therapy conversations: 3,512
- Multilingual greetings (15+ languages): 1,920
- Total: 5,432 conversations

**Split:**
- Training: 2,000 (36.8%)
- Validation: 1,716 (31.6%)
- Test: 1,716 (31.6%)

**Topics:** Anxiety, depression, stress, sleep problems, relationships, coping strategies

**Preprocessing:**
- Formatted with 'Patient' and 'Therapist' labels
- Added 200+ mental health keywords
- Tokenized with GPT-2 tokenizer (max 150 tokens)

## Model Architecture

**Base Model:** GPT-2 (124,439,808 parameters)

**Components:**
- Tokenizer: GPT-2 (50,257 vocabulary)
- Embedding: 768-dimensional
- Transformer: 12 layers with multi-head attention
- Output: Language modeling head

**Training Configuration:**

**Features:**
- Multilingual greeting support (15+ languages)
- Context-aware responses
- Mental health keyword detection (200+ keywords)
- Off-topic redirection
- Gradio web interface

## Technical Challenges

**Challenge 1: CPU Training**
- Problem: Extremely slow (hours per epoch)
- Solution: Abandoned CPU approach

**Challenge 2: Local Machine**
- Problem: HP laptop insufficient RAM/GPU
- Solution: Migrated to cloud

**Challenge 3: Final Solution**
- Platform: Google Colab with Tesla T4 GPU
- Result: Training completed in 1h 18m 52s
- Learning: GPU acceleration essential for LLMs

## Training Process

**Progress:**

| Step | Training Loss | Validation Loss |
|------|---------------|-----------------|
| 200  | 3.0600        | 2.8893         |
| 2000 | 2.0200        | 2.4000         |
| 4000 | 1.7500        | 2.3500         |
| 6000 | 1.5800        | 2.3300         |
| 8000 | 1.4500        | 2.3200         |
| 10000| 1.3600        | 2.3177         |

**Hardware:** Tesla T4 GPU (Google Colab)  
**Duration:** 1 hour 18 minutes 52 seconds

## Results & Evaluation

**Final Metrics:**
- Training Loss: 1.7941 (55.6% improvement from 3.06)
- Validation Loss: 2.3177 (19.7% improvement from 2.89)
- Perplexity: 10.0866

**Test Results:**

| Test Category | Status | Accuracy |
|---------------|--------|----------|
| Multilingual Greetings | ✓ Pass | 100% |
| Off-Topic Detection | ✓ Pass | 100% |
| Mental Health Responses | ✓ Pass | Functional |
| Context Memory | ✓ Pass | Working |

**Sample Tests:**
Test 1: "Hello"
Output: "Hello! I'm here to support you. How can I help you today?"
Status: ✓ Pass

Test 2: "What's the weather like?"
Output: "I'm a mental health support chatbot. I'm here to help with feelings, stress, anxiety, and emotional well-being. How are you feeling today?"
Status: ✓ Pass

Test 3: "I've been feeling anxious lately"
Output: [Generated empathetic response about anxiety]
Status: ✓ Pass

**Key Observations:**
- Consistent improvement throughout training
- No significant overfitting
- Model converged successfully
- Good prediction confidence (perplexity 10.09)

## How to Run

**Prerequisites:**
- Python 3.8+
- Google Colab account
- GPU runtime (T4)

**Steps:**

1. Clone repository:
```bash
git clone https://github.com/Leslyndizeye/mental-health-chatbot.git
Open `mental_health_chatbot.ipynb` in Google Colab
Enable GPU: Runtime → Change runtime type → GPU (T4)
Run all cells
Gradio interface launches with public link
