# 📚 Research-GPT: A Retrieval-Augmented Generation (RAG) System

Research-GPT is a powerful document-based Q&A system built using **Streamlit, FAISS, and Ollama LLMs**. It allows users to query a collection of research papers (PDFs), retrieve relevant information, and get AI-generated answers in a structured and context-aware format.

## 🚀 Features
- 📄 **Automatic PDF Processing** – Upload research papers or any text-heavy PDFs.
- 🔍 **Intelligent Document Search** – Uses **FAISS** to retrieve the most relevant chunks.
- 🧠 **AI-Powered Answers** – Utilizes **Ollama LLMs (Llama 3.2:1B)** to generate responses.
- 🎯 **Optimized for Research** – Provides well-structured bullet-point answers from context only.
- 🖥️ **Interactive UI with Streamlit** – Simple and user-friendly web-based interface.

## 📂 Project Structure
```
Research-GPT/
│── rag-dataset/            # Folder containing research PDFs
│── app.py                  # Main application script (Streamlit UI + RAG Pipeline)
│── requirements.txt        # List of dependencies
│── README.md               # Project documentation (this file)
```

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```sh
git clone https://github.com/your-repo/research-gpt.git
cd research-gpt
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv rag
source rag/bin/activate  # For Mac/Linux
rag\Scripts\activate     # For Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Start the Ollama Server
Download and install [Ollama](https://ollama.com), then run:
```sh
ollama serve
```

### 5️⃣ Run the Application
```sh
streamlit run app.py
```

## 🏗️ How It Works

1️⃣ **Load PDFs:** The system scans the `rag-dataset/` folder and extracts text.

2️⃣ **Split Text into Chunks:** Documents are split into smaller, meaningful sections.

3️⃣ **Embed and Index Text:** FAISS stores and indexes document embeddings for efficient retrieval.

4️⃣ **User Queries:** When a user asks a question, the system retrieves the most relevant text chunks.

5️⃣ **AI-Powered Answer Generation:** The retrieved chunks are passed to an **LLM (Llama 3.2:1B)** to generate a response.

6️⃣ **Display Results:** The answer is displayed in a clean and structured format via Streamlit.

## 🎯 Example Query
**User Input:** _"What is the impact of AI on cybersecurity?"_

**Response:**
✅ AI is used in cybersecurity for:
- Threat detection and anomaly identification.
- Automated response to security incidents.
- Enhancing fraud detection using machine learning.

## ⚡ Technologies Used
- **Python** – Core language for backend processing.
- **Streamlit** – Interactive and user-friendly UI.
- **FAISS** – Efficient similarity search for document retrieval.
- **Ollama LLMs (Llama 3.2:1B)** – AI model for response generation.
- **PyMuPDF** – PDF text extraction and processing.

## 🌟 Future Enhancements
- 🔗 **Web Scraper** – Fetch real-time articles for enhanced knowledge retrieval.
- 🎙️ **Voice Input** – Allow users to speak queries.
- 🌍 **Multi-Language Support** – Process and answer questions in various languages.

## 🤝 Contributing
Want to improve Research-GPT? Feel free to fork the repository and submit pull requests! 🚀

## 📜 License
This project is licensed under the **MIT License**.

---
💡 _Built with ❤️ by [Your Name]_

