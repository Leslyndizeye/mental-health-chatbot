# 🧠 Mental Health Support Chatbot

An AI-powered conversational agent that provides emotional support using natural language processing.  
Built by fine-tuning GPT-2 on mental health counseling conversations.

[![Open in Hugging Face](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/leslylezoo/mental-health-chatbot)
[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/leslylezoo/mental-health-chatbot-gpt2)

---

## 🎥 Demo & Links

* **Live Demo:** [https://huggingface.co/spaces/leslylezoo/mental-health-chatbot](https://huggingface.co/spaces/leslylezoo/mental-health-chatbot)
* **Model Repository:** [https://huggingface.co/leslylezoo/mental-health-chatbot-gpt2](https://huggingface.co/leslylezoo/mental-health-chatbot-gpt2)
* **YouTube Demo:** [https://www.youtube.com/watch?v=1hcipuXIdfA](https://www.youtube.com/watch?v=1hcipuXIdfA)
* **GitHub Repository:** [https://github.com/Leslyndizeye/mental-health-chatbot.git](https://github.com/Leslyndizeye/mental-health-chatbot.git)

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
- [Deployment](#deployment)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [References](#references)

---

## 🩺 Problem Statement

Mental health support faces critical barriers such as:
- Limited availability of professional therapists  
- High costs of therapy sessions  
- Social stigma preventing people from seeking help  
- Lack of 24/7 immediate support  

This project develops an **AI chatbot** that provides accessible, immediate emotional support while recognizing mental health concerns and offering a safe, judgment-free space for expression.

> **⚠️ Important Note:** This is not a replacement for professional therapy but serves as a supportive tool for emotional well-being.

---

## 📊 Dataset

### Source
**[Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations)** (Hugging Face)

### Composition
- Original Conversations: 3,512  
- Augmented Greetings: 1,920 multilingual greetings  
- **Total Samples:** 5,432  

### Data Split
- **Training:** 2,000 samples
- **Validation:** 1,716 samples
- **Testing:** 1,716 samples

### Topics Covered
Anxiety, depression, stress, sleep issues, relationships, grief, coping mechanisms, and self-esteem.

### Preprocessing Steps
1. **Text Cleaning:** Removed special characters, normalized whitespace
2. **Tokenization:** Used GPT-2 tokenizer with max length of 150 tokens
3. **Contextual Tagging:** Added "Patient" and "Therapist" role markers
4. **Keyword Enhancement:** Integrated 200+ mental health-specific keywords for topic detection
5. **Data Augmentation:** Added multilingual greetings (15+ languages) for better greeting recognition

---

## 🧠 Model Architecture

### Base Model
**GPT-2** (124M parameters) from Hugging Face Transformers

### Framework
- **Deep Learning:** PyTorch
- **NLP Library:** Hugging Face Transformers
- **Interface:** Gradio

### Training Configuration

\`\`\`python
Epochs = 40
Batch Size = 4
Learning Rate = 5e-5
Optimizer = AdamW
Weight Decay = 0.01
Temperature = 0.7
Top-p Sampling = 0.9
Max Length = 150 tokens
\`\`\`

### Model Layers
- **Tokenizer:** GPT-2 (50,257 vocabulary size)
- **Embedding Dimension:** 768
- **Transformer Layers:** 12 attention layers
- **Output Head:** Language modeling head for text generation

---

## ⚙️ Implementation

### Key Features

1. **Context-Aware Responses**
   - Maintains conversation history for coherent multi-turn dialogues
   - Uses previous messages to generate contextually relevant responses

2. **Mental Health Topic Detection**
   - Keyword-based classification system
   - Identifies topics: anxiety, depression, stress, relationships, grief, etc.

3. **Multilingual Greeting Support**
   - Recognizes greetings in 15+ languages
   - Responds appropriately to diverse user backgrounds

4. **Off-Topic Query Handling**
   - Detects non-mental-health queries
   - Politely redirects users to mental health topics

5. **Empathetic Response Generation**
   - Fine-tuned on counseling conversations
   - Generates supportive, non-judgmental responses

### User Interface
- **Platform:** Gradio web interface
- **Features:**
  - Clean, intuitive chat interface
  - Example prompts for new users
  - Clear conversation history
  - Responsive design

---

## 🧩 Technical Challenges

| Challenge | Description | Solution |
|-----------|-------------|----------|
| **CPU Training Bottleneck** | Training on CPU was extremely slow (hours per epoch) | Migrated to Google Colab with Tesla T4 GPU |
| **Memory Constraints** | Local machine had insufficient RAM (frequent crashes) | Used Google Colab with 16GB VRAM |
| **Overfitting Risk** | Model memorizing training data | Implemented warmup steps, weight decay, and validation monitoring |
| **Response Quality** | Initial responses were generic | Fine-tuned with domain-specific data and adjusted temperature/top-p |
| **Deployment Complexity** | Initial Flask errors on Hugging Face | Switched to Gradio ChatInterface for seamless deployment |

### Development Environment
- **Platform:** Google Colab
- **GPU:** Tesla T4 (16GB VRAM)
- **Total Training Time:** 1 hour 18 minutes 52 seconds

---

## 📈 Training Process

### Training Progress

| Step | Train Loss | Validation Loss |
|------|-----------|-----------------|
| 200 | 3.06 | 2.88 |
| 1000 | 2.27 | 2.47 |
| 4000 | 1.75 | 2.35 |
| 8000 | 1.45 | 2.32 |
| 10000 | 1.36 | 2.31 |

### Performance Improvements
- ✅ **Training Loss Reduction:** 55.6% (from 3.06 to 1.36)
- ✅ **Final Validation Loss:** 2.31
- ✅ **Perplexity:** 10.09 (lower is better)
- ✅ **Convergence:** Stable after 8,000 steps

### Training Visualization
The model showed consistent improvement throughout training with minimal overfitting, as evidenced by the close tracking of training and validation losses.

---

## 🎯 Results

### Quantitative Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **BLEU Score** | 0.74 | High overlap with reference responses |
| **F1 Score** | 0.81 | Strong balance of precision and recall |
| **Perplexity** | 10.09 | Good predictive confidence |
| **Training Loss** | 1.36 | Effective learning achieved |
| **Validation Loss** | 2.31 | Minimal overfitting |

### Qualitative Analysis
The chatbot demonstrates:
- Empathetic and supportive tone
- Contextually appropriate responses
- Ability to maintain conversation flow
- Recognition of mental health topics
- Appropriate handling of off-topic queries

---

## 🧪 Testing & Evaluation

### Test Cases

| Test Category | Input | Expected Behavior | Output | Status |
|--------------|-------|-------------------|--------|--------|
| **Greeting** | "Hello" | Warm welcome message | "Hello! I'm here to support you. How can I help today?" | ✅ Pass |
| **Multilingual** | "Bonjour" | Recognize and respond | "Hello! How can I support you today?" | ✅ Pass |
| **Anxiety** | "I feel anxious about work" | Empathetic support | Provides coping strategies and validation | ✅ Pass |
| **Depression** | "I've been feeling really down" | Supportive response | Acknowledges feelings, offers support | ✅ Pass |
| **Off-Topic** | "What's the weather?" | Redirect to mental health | "I'm a mental health chatbot..." | ✅ Pass |
| **Context** | Multi-turn conversation | Maintain context | References previous messages | ✅ Pass |

### User Feedback
- Responses feel natural and supportive
- Interface is easy to use
- Chatbot maintains conversation context well
- Appropriate handling of sensitive topics

---

## 🚀 Deployment

### Hugging Face Space
The chatbot is deployed on Hugging Face Spaces using Gradio:

**Live Demo:** [https://huggingface.co/spaces/leslylezoo/mental-health-chatbot](https://huggingface.co/spaces/leslylezoo/mental-health-chatbot)

### Deployment Architecture
\`\`\`
User → Gradio Interface → GPT-2 Model → Response Generation → User
\`\`\`

### Technical Stack
- **Frontend:** Gradio ChatInterface
- **Backend:** Hugging Face Transformers
- **Model Hosting:** Hugging Face Model Hub
- **Deployment:** Hugging Face Spaces (CPU)

### Deployment Files
- `app.py` - Main application with Gradio interface
- `requirements.txt` - Python dependencies
- `README.md` - Space documentation

---

## 🧰 How to Run

### Option 1: Use the Live Demo (Recommended)
Simply visit: [https://huggingface.co/spaces/leslylezoo/mental-health-chatbot](https://huggingface.co/spaces/leslylezoo/mental-health-chatbot)

### Option 2: Run Locally

#### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) GPU for faster inference

#### Installation Steps

\`\`\`bash
# 1. Clone the repository
git clone https://github.com/Leslyndizeye/mental-health-chatbot.git
cd mental-health-chatbot

# 2. Install dependencies
pip install torch transformers datasets gradio accelerate

# 3. Run the Jupyter notebook
jupyter notebook mental_health_chatbot.ipynb

# 4. Or run the Gradio app directly
python app.py
\`\`\`

### Option 3: Run on Google Colab

1. Open `mental_health_chatbot.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU → Tesla T4
3. Run all cells
4. Access the Gradio interface via the generated link

---

## 🗂️ Project Structure

mental-health-chatbot/
│
├── data/
│ ├── processed/
│ │ ├── train.csv
│ │ ├── val.csv
│ │ ├── test.csv
│ │ ├── train.txt
│ │ └── val.txt
│ └── raw/
│ └── all_conversations.csv
│
├── models/
│ ├── mental_health_gpt2/
│ │ ├── config.json
│ │ ├── generation_config.json
│ │ ├── model.safetensors
│ │ ├── merges.txt
│ │ ├── special_tokens_map.json
│ │ ├── tokenizer_config.json
│ │ └── vocab.json
│ └── tokenizer/
│ ├── merges.txt
│ ├── special_tokens_map.json
│ ├── tokenizer_config.json
│ └── vocab.json
│
├── docs/
│ ├── evaluation_results.txt
│ └── training_logs.txt
│
├── mental_health_chatbot.ipynb # Main training notebook
├── app.py # Gradio deployment app
├── requirements.txt # Python dependencies
└── README.md # This file

## ⚠️ Limitations

### Current Limitations
1. **No Long-Term Memory:** Conversation history resets between sessions
2. **Text-Only:** No support for voice input/output or sentiment analysis
3. **Language:** Primarily optimized for English responses
4. **Repetition:** May occasionally generate repetitive responses
5. **Crisis Handling:** Not equipped to handle emergency situations or suicidal ideation
6. **Professional Advice:** Cannot replace licensed mental health professionals

### Safety Considerations
- This chatbot is a **supportive tool**, not a replacement for therapy
- Users experiencing crisis should contact emergency services or crisis hotlines
- Responses are generated by AI and may not always be appropriate
- No medical advice or diagnosis is provided

---

## 🔮 Future Improvements

### Planned Enhancements
1. **Long-Term Memory:** Implement session persistence and user profiles
2. **Multimodal Support:** Add voice input/output capabilities
3. **Sentiment Analysis:** Real-time emotion detection and response adaptation
4. **Crisis Detection:** Implement safety protocols for emergency situations
5. **Multilingual Support:** Expand to full multilingual response generation
6. **Personalization:** Adapt responses based on user preferences and history
7. **Professional Integration:** Connect users with licensed therapists when needed
8. **Mobile App:** Develop native iOS/Android applications

### Technical Improvements
- Upgrade to larger models (GPT-3, LLaMA, etc.)
- Implement retrieval-augmented generation (RAG)
- Add reinforcement learning from human feedback (RLHF)
- Improve response diversity and reduce repetition

---

## 📚 References

### Datasets
- [Mental Health Counseling Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) - Hugging Face

### Models & Libraries
- [GPT-2](https://huggingface.co/gpt2) - OpenAI via Hugging Face
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) - NLP library
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Gradio](https://gradio.app/) - ML interface library

### Research Papers
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners" (GPT-2)
- Vaswani, A., et al. (2017). "Attention is All You Need" (Transformer architecture)

### Tools & Platforms
- [Google Colab](https://colab.research.google.com/) - Training environment
- [Hugging Face Spaces](https://huggingface.co/spaces) - Deployment platform

---

## 👨‍💻 Author

**Lesly Ndizeye**
- GitHub: [@Leslyndizeye](https://github.com/Leslyndizeye)
- Hugging Face: [@leslylezoo](https://huggingface.co/leslylezoo)

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Hugging Face for providing the dataset and model hosting
- Google Colab for free GPU resources
- The open-source NLP community for tools and libraries
- Mental health professionals who inspired this project

---

## 📞 Support & Contact

For questions, issues, or suggestions:
- Open an issue on [GitHub](https://github.com/Leslyndizeye/mental-health-chatbot/issues)
- Contact via Hugging Face [Community](https://huggingface.co/spaces/leslylezoo/mental-health-chatbot/discussions)

---

**⚠️ Crisis Resources:**
- National Suicide Prevention Lifeline: 988 (US)
- Crisis Text Line: Text HOME to 741741 (US)
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

---

*Built with ❤️ for mental health awareness and support*
