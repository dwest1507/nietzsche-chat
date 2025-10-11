# 🚀 Setup Guide for Nietzsche Chatbot

Follow these steps to get your Nietzsche chatbot up and running!

## Step 1: Install Python Dependencies

Open your terminal and run:

```bash
pip install -r requirements.txt
```

This will install:
- Streamlit (web interface)
- LangChain (RAG framework)
- FAISS (vector database)
- Groq API client
- HuggingFace embeddings

**Estimated time:** 2-3 minutes

## Step 2: Get Your Free Groq API Key

1. Visit: **https://console.groq.com/**
2. Click "Sign Up" or "Log In"
3. Once logged in, go to "API Keys" section
4. Click "Create API Key"
5. Give it a name (e.g., "nietzsche-chatbot")
6. Copy the API key (you won't see it again!)

**Cost:** FREE ✅
- 14,400 requests per day
- 30 requests per minute
- No credit card required

## Step 3: Configure Your API Key

### On Windows:
```bash
mkdir .streamlit
copy .streamlit\secrets.toml.example .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```

### On Mac/Linux:
```bash
mkdir -p .streamlit
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
nano .streamlit/secrets.toml
```

Then edit the file and replace `your_groq_api_key_here` with your actual API key:

```toml
GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxxxxxxx"
```

**Important:** Don't share this file or commit it to GitHub!

## Step 4: Build the Vector Store

This is a one-time process that indexes all of Nietzsche's works:

```bash
python scripts/build_vectorstore.py
```

What happens:
- ✅ Loads 19 Nietzsche texts (~3MB of text)
- ✅ Splits into ~15,000 chunks
- ✅ Creates embeddings (uses your CPU, not GPU)
- ✅ Builds FAISS vector database
- ✅ Saves to `vectorstore/` folder

**Estimated time:** 2-5 minutes (depending on your machine)

You'll see progress output like:
```
Loading Nietzsche texts...
Found 19 texts
  ✓ Loaded Thus Spake Zarathustra
  ✓ Loaded Beyond Good And Evil
  ...
✓ Vector store successfully saved to vectorstore/
```

## Step 5: Run the App!

```bash
streamlit run app.py
```

Your browser should automatically open to `http://localhost:8501`

If it doesn't, manually visit that URL.

## Step 6: Start Chatting!

Try these questions:
- "What is the Übermensch?"
- "Why do you criticize Christianity?"
- "What is the Will to Power?"
- "What would you think about modern social media?"

## 🐛 Troubleshooting

### Error: "Vector store not found"
→ Run `python scripts/build_vectorstore.py` first

### Error: "Groq API key not found"
→ Check that `.streamlit/secrets.toml` exists and has your API key

### Error: "No module named 'langchain'"
→ Run `pip install -r requirements.txt` again

### Error: "Rate limit exceeded"
→ You've hit Groq's free tier limit. Wait a bit or try tomorrow.

### Slow response times
→ Normal! First response loads models. Subsequent ones are faster.

### "Could not find file" errors
→ Make sure you're running from the project root directory

## 🌐 Deploying to the Internet

Want to share your chatbot with others? See the **Deployment to Streamlit Community Cloud** section in the README.md.

## 💡 Tips

1. **Chat History**: Click "Clear Chat History" in the sidebar to start fresh
2. **Source Passages**: Expand "View source passages" to see where responses come from
3. **Temperature**: Lower temperature (0.5) = more consistent, Higher (0.9) = more creative
4. **Experiment**: Try different questions and see how Nietzsche responds!

## 🎉 You're All Set!

Enjoy your philosophical conversations with Nietzsche!

---

*"You must have chaos within you to give birth to a dancing star."* - Friedrich Nietzsche

