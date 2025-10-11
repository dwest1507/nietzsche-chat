# 💭 Nietzsche Chatbot

A conversational AI that embodies Friedrich Nietzsche's philosophical voice and ideas using Retrieval-Augmented Generation (RAG).

Chat with Nietzsche about philosophy, morality, religion, human nature, and more. The bot draws from 19 of his complete works to provide responses grounded in his actual writings and philosophical positions.

## ✨ Features

- **RAG-Powered**: Retrieves relevant passages from Nietzsche's works to ground responses
- **19 Complete Works**: Including *Thus Spake Zarathustra*, *Beyond Good and Evil*, *The Genealogy of Morals*, and more
- **Authentic Voice**: Carefully crafted prompts to capture Nietzsche's bold, aphoristic style
- **Free to Use**: Powered by Groq's free Llama 3.1 API (no cost!)
- **Source Citations**: View the exact passages used to generate each response

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A free Groq API key ([get one here](https://console.groq.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/nietzsche-chat.git
   cd nietzsche-chat
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free Groq API key**
   - Go to [https://console.groq.com/](https://console.groq.com/)
   - Sign up for a free account
   - Create an API key from the dashboard
   - Copy the API key

4. **Configure your API key**
   ```bash
   # Create the secrets file
   mkdir -p .streamlit
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   
   # Edit .streamlit/secrets.toml and add your API key:
   # GROQ_API_KEY = "your_actual_api_key_here"
   ```

5. **Build the vector store** (one-time setup)
   ```bash
   python scripts/build_vectorstore.py
   ```
   This will:
   - Load all 19 Nietzsche texts
   - Split them into chunks
   - Create embeddings
   - Build and save a FAISS vector store
   - Takes ~2-5 minutes depending on your machine

6. **Run the app**
   ```bash
   streamlit run app.py
   ```

7. **Open your browser** to `http://localhost:8501` and start chatting!

## 📖 How It Works

1. **User asks a question** → "What is the Übermensch?"
2. **Vector search** → The system searches through Nietzsche's works for relevant passages
3. **Context retrieval** → Top 4 most relevant passages are retrieved
4. **LLM generation** → Llama 3.1 generates a response in Nietzsche's voice using these passages
5. **Response** → You get an authentic Nietzschean answer with source citations

## 🎯 Usage Tips

### Good Questions to Ask

- "What is the Will to Power?"
- "Why do you criticize Christianity?"
- "What is the difference between master and slave morality?"
- "Explain the concept of eternal recurrence"
- "What would you think about modern social media?"

### Conversation Tips

- Challenge Nietzsche's ideas and engage in philosophical debate
- Ask for clarification on specific concepts
- Request his views on modern topics
- Ask about specific works or passages

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **LLM**: Groq API (Llama 3.1 8B)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **RAG Framework**: LangChain

## 📚 Included Works

The chatbot has access to these 19 complete works by Nietzsche:

1. Thus Spake Zarathustra
2. Beyond Good and Evil
3. The Genealogy of Morals
4. The Antichrist
5. Ecce Homo
6. The Birth of Tragedy
7. Human, All-Too-Human
8. The Joyful Wisdom
9. The Twilight of the Idols
10. The Dawn of Day
11. The Will to Power (Books I-II)
12. The Will to Power (Books III-IV)
13. Thoughts out of Season (Part I)
14. Thoughts out of Season (Part II)
15. The Case of Wagner
16. Early Greek Philosophy and Other Essays
17. Homer and Classical Philology
18. On the Future of Our Educational Institutions
19. We Philologists

All texts are from public domain translations available on Project Gutenberg.

## 🌐 Deployment to Streamlit Community Cloud

### Option 1: Commit Vector Store (Recommended)

1. Remove `vectorstore/` from `.gitignore` if present
2. Run `python scripts/build_vectorstore.py` locally
3. Commit and push the `vectorstore/` directory to your repo
4. Deploy to Streamlit Cloud:
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Connect your GitHub repo
   - Add your `GROQ_API_KEY` to Secrets (in app settings)
   - Deploy!

### Option 2: Build on First Run

If you don't want to commit the vector store:

1. Keep `vectorstore/` in `.gitignore`
2. Modify `app.py` to build the vector store on first run if it doesn't exist
3. Note: First load will be slower (~2-5 minutes)

### Adding Secrets to Streamlit Cloud

1. Go to your app settings on Streamlit Cloud
2. Navigate to "Secrets"
3. Add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```

## 💰 Cost & Rate Limits

This project uses **Groq's free tier**:
- ✅ 14,400 requests per day
- ✅ 30 requests per minute
- ✅ Fast inference (~500 tokens/sec)
- ✅ **Completely free** for personal/demo use

Perfect for a personal chatbot or portfolio project!

## 🤝 Contributing

Contributions are welcome! Some ideas:
- Add more Nietzsche works
- Improve the personality prompt
- Add conversation history export
- Implement multi-language support
- Add more advanced RAG techniques

## 📄 License

This project is open source. The texts are from public domain translations.

## 🙏 Acknowledgments

- Texts from [Project Gutenberg](https://www.gutenberg.org/)
- Powered by [Groq](https://groq.com/) for fast LLM inference
- Built with [Streamlit](https://streamlit.io/) and [LangChain](https://langchain.com/)

## 📧 Questions or Issues?

Open an issue on GitHub or reach out!

---

*"And those who were seen dancing were thought to be insane by those who could not hear the music."* - Friedrich Nietzsche
