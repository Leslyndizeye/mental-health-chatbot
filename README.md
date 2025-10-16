## 🎥 Demo & Repository

* **YouTube Demo:** [https://www.youtube.com/watch?v=1hcipuXIdfA](https://www.youtube.com/watch?v=1hcipuXIdfA)
* **GitHub Repository:** [https://github.com/Leslyndizeye/mental-health-chatbot.git](https://github.com/Leslyndizeye/mental-health-chatbot.git)
  
# 🧠 Mental Health Support Chatbot

An AI-powered conversational agent that provides emotional support using natural language processing.  
Built by fine-tuning GPT-2 on mental health counseling conversations.
---

## 📚 Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Implementation](#implementation)
- [Technical Challenges](#technical-challenges)
- [Training Process](#training-process)
- [Results](#results)
- [Testing & Evaluation](#testing--evaluation)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [References](#references)

---

## 🩺 Problem Statement

Mental health support faces critical barriers such as:
- Limited availability of professional therapists  
- High costs of therapy sessions  
- Social stigma preventing people from seeking help  
- Lack of 24/7 immediate support  

This project develops an **AI chatbot** that provides accessible, immediate emotional support while recognizing mental health concerns and offering a safe, judgment-free space for expression.

> **Note:** This is not a replacement for professional therapy but serves as a supportive tool.

---

## 📊 Dataset

### Source
**[Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)** (Hugging Face)

### Composition
- Original Conversations: 3,512  
- Augmented Greetings: 1,920 multilingual greetings  
- **Total Samples:** 5,432  

### Data Split
- **Training:** 2,000  
- **Validation:** 1,716  
- **Testing:** 1,716  

### Topics Covered
Anxiety, depression, stress, sleep, relationships, grief, coping, and self-esteem.

### Preprocessing
- Cleaned and normalized text  
- Tokenized with GPT-2 tokenizer (max length: 150)  
- Contextual tagging with "Patient" and "Therapist" roles  
- Added 200+ mental health keywords for topic detection  

---

## 🧠 Model Architecture

- **Base Model:** GPT-2 (124M parameters)  
- **Framework:** Hugging Face Transformers + PyTorch  

**Configuration**

Epochs = 40
Batch Size = 4
Learning Rate = 5e-5
Optimizer = AdamW
Weight Decay = 0.01
Temperature = 0.7
Top-p Sampling = 0.9

**Layers**

* Tokenizer: GPT-2 (50,257 vocab)
* Embedding: 768D
* Transformer: 12 layers
* Output: Language modeling head

---

## ⚙️ Implementation

### Key Features

1. **Multilingual Greeting Support (15+ languages)**
2. **Context-Aware Responses** using conversation history
3. **Topic Detection** via keyword classification
4. **Off-Topic Redirection** for non-mental-health queries
5. **Empathetic Text Generation**

### Interface

* Built using **Gradio**
* Includes **typing effect**, **clear input**, and **new chat** buttons
* Example prompts for beginners

---

## 🧩 Technical Challenges

| Challenge          | Description                      | Solution                      |
| ------------------ | -------------------------------- | ----------------------------- |
| CPU Training       | Extremely slow (hours per epoch) | Moved to GPU runtime          |
| Low RAM on Laptop  | Frequent memory crashes          | Used Google Colab             |
| Training Stability | Overfitting risk                 | Added warmup + regularization |

**Final Platform:** Google Colab (Tesla T4 GPU, 16GB VRAM)
**Training Time:** 1 hour 18 minutes 52 seconds

---

## 📈 Training Process

| Step  | Train Loss | Val Loss |
| ----- | ---------- | -------- |
| 200   | 3.06       | 2.88     |
| 1000  | 2.27       | 2.47     |
| 4000  | 1.75       | 2.35     |
| 8000  | 1.45       | 2.32     |
| 10000 | 1.36       | 2.31     |

✅ **Improvement:** 55.6% training loss reduction
✅ **Validation Loss:** 2.31
✅ **Perplexity:** 10.09

---

## 🧪 Testing & Evaluation

| Test      | Input                 | Output                                                  | Status |
| --------- | --------------------- | ------------------------------------------------------- | ------ |
| Greeting  | “Hello”               | “Hello! I'm here to support you. How can I help today?” | ✅      |
| Off-topic | “What's the weather?” | “I'm a mental health chatbot…”                          | ✅      |
| Anxiety   | “I feel anxious.”     | Empathetic support response                             | ✅      |
| Context   | “Hello” → “I'm good.” | “That's great! How are you feeling today?”              | ✅      |

**Metrics**

* BLEU: 0.74
* F1-score: 0.81
* Perplexity: 10.09

---

## 🧰 How to Run

### Prerequisites

* Python 3.8+
* GPU Runtime (Google Colab recommended) for me i used t4 gpu from google Colab

### Steps

```bash
# 1. Clone repo
git clone https://github.com/Leslyndizeye/mental-health-chatbot.git

# 2. Install dependencies
pip install torch transformers datasets gradio accelerate

# 3. Run notebook
Open `mental_health_chatbot.ipynb` in Google Colab

# 4. Enable GPU
Runtime → Change runtime type → GPU → Tesla T4

# 5. Launch Chatbot
Run all cells → Access Gradio interface link

---

## 🗂️ Project Structure


|-- data
|   |-- processed
|   |   |-- test.csv
|   |   |-- train.csv
|   |   |-- train.txt
|   |   |-- val.csv
|   |   `-- val.txt
|   `-- raw
|       `-- all_conversations.csv
|-- docs
|   |-- evaluation_results.txt
|   `-- training_logs.txt
|-- mental_health_chatbot.ipynb
|-- models
|   |-- mental_health_gpt2
|   |   |-- config.json
|   |   |-- generation_config.json
|   |   |-- merges.txt
|   |   |-- special_tokens_map.json
|   |   |-- tokenizer_config.json
|   |   `-- vocab.json
|   `-- tokenizer
|       |-- merges.txt
|       |-- special_tokens_map.json
|       |-- tokenizer_config.json
|       `-- vocab.json
|-- README.md


---

## ⚠️ Limitations

* Responses may occasionally repeat
* No long-term memory between sessions
* Only text-based (no voice/sentiment)
* Cannot handle emergencies or suicidal ideation
* Primary language: English

---
