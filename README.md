# Mental Health Support Chatbot

An AI-powered conversational agent designed to provide emotional support and mental health assistance using natural language processing. Built by fine-tuning GPT-2 on specialized mental health counseling conversations.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)
![Transformers](https://img.shields.io/badge/Transformers-4.x-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training Results](#training-results)
- [Demo](#demo)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

Mental health support is increasingly important, but access to professional help can be limited due to cost, availability, or stigma. This project creates an accessible, judgment-free AI assistant that provides immediate emotional support 24/7.

**Note:** This chatbot is not a replacement for professional therapy but serves as a supportive tool for those who need someone to talk to.

## Features

- **Multilingual Greeting Support**: Recognizes and responds to greetings in 15+ languages (English, Spanish, French, German, Italian, Portuguese, Hindi, Arabic, Chinese, Japanese, Korean, Russian, and more)
- **Context-Aware Conversations**: Remembers conversation context and provides relevant follow-up responses
- **Mental Health Topic Detection**: Uses 200+ keywords to identify mental health-related queries
- **Off-Topic Detection**: Intelligently redirects non-mental-health queries back to supportive topics
- **Empathetic Responses**: Generates compassionate, supportive responses for emotional concerns
- **Interactive Interface**: User-friendly Gradio interface with typing animation effects
- **Real-time Interaction**: Instant responses with clear input and new chat functionality

## Dataset

### Source
- **Primary Dataset**: [Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) from Hugging Face
- **Size**: 3,512 real therapy conversations between patients and therapists

### Data Augmentation
- Added 1,920 multilingual greeting examples (duplicated 30x each for effective learning)
- Implemented 200+ mental health keywords for topic detection
- Created context-aware follow-up response system

### Final Dataset Statistics
- **Total Conversations**: 5,432
- **Training Set**: 2,000 samples (37%)
- **Validation Set**: 1,716 samples (31.5%)
- **Test Set**: 1,716 samples (31.5%)

## Model Architecture

### Base Model
- **Model**: GPT-2 (Generative Pre-trained Transformer 2)
- **Parameters**: 124,439,808 (124 million)
- **Architecture**: Transformer with causal language modeling
- **Framework**: Hugging Face Transformers + PyTorch

### Training Configuration
```python
Training Epochs: 40
Total Steps: 10,000
Batch Size: 4 (with gradient accumulation)
Learning Rate: 5e-5 (with warmup)
Optimizer: AdamW
Max Length: 150 tokens
Temperature: 0.7
Top-p Sampling: 0.9
### Hardware Requirements

- **GPU**: Tesla T4 (16GB VRAM) via Google Colab
- **Training Time**: 1 hour 18 minutes 52 seconds
- **Note**: CPU training and local laptop (HP) attempts failed due to insufficient computational resources
## Installation



### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- Google Colab account (for training)
