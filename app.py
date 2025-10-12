"""
Nietzsche Chatbot - A RAG-powered conversational AI embodying Friedrich Nietzsche
"""
import streamlit as st
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from typing import List, Any
from pydantic import Field, ConfigDict
import logging
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="Chat with Nietzsche",
    page_icon="📚",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    /* Main Layout */
    .stTitle {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding-bottom: 1rem;
    }
    
    /* Header Section */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .subtitle {
        text-align: center;
        font-style: italic;
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        margin: 0;
    }
    
    /* Chat Messages */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* User message styling */
    [data-testid="stChatMessageContent"] {
        background-color: transparent;
    }
    
    /* Source citations card styling */
    .source-card {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 0.75rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .source-title {
        color: #667eea;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .source-content {
        color: #333;
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] * {
        color: #333 !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #667eea !important;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #764ba2 !important;
        font-size: 1.25rem;
        margin-top: 1.25rem;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] a {
        color: #667eea !important;
    }
    
    /* Clear chat button */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-top: 2px solid #e0e0e0;
        padding-top: 1rem;
    }
    
    /* Typography improvements */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        line-height: 1.6;
    }
    
    p {
        margin-bottom: 1rem;
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-text {
        animation: pulse 1.5s ease-in-out infinite;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .subtitle {
            font-size: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Enhanced header with gradient background
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">💭 Chat with Friedrich Nietzsche</h1>
        <p class="subtitle">"Thus spake Zarathustra... and now he answers your questions."</p>
    </div>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

@st.cache_resource
def load_vectorstore():
    """Load the FAISS vector store from disk."""
    vectorstore_path = Path("vectorstore")
    
    if not vectorstore_path.exists():
        st.error("""
        ⚠️ Vector store not found! 
        
        Please run the following command first to build the vector store:
        ```
        python scripts/build_vectorstore.py
        ```
        """)
        st.stop()
    
    # Phase 3: Using better embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vectorstore = FAISS.load_local(
        str(vectorstore_path),
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    return vectorstore

@st.cache_resource
def initialize_chain(_vectorstore):
    """Initialize the conversational retrieval chain with Phase 2 enhancements."""
    
    # Check for API key
    if "GROQ_API_KEY" not in st.secrets:
        st.error("""
        ⚠️ Groq API key not found!
        
        Please add your API key to `.streamlit/secrets.toml`:
        ```toml
        GROQ_API_KEY = "your_api_key_here"
        ```
        
        Get a free API key at: https://console.groq.com/
        """)
        st.stop()
    
    # Initialize Groq LLM with low temperature for faithfulness to source material
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0.3,  # Reduced from 0.7 for more faithful, less creative responses
        max_tokens=1024
    )
    
    # Suppress MultiQueryRetriever logging (optional)
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.ERROR)
    
    # Phase 3: Query Expansion Helper
    def expand_query(question):
        """Expand philosophical concepts to related terms."""
        expansions = {
            "happiness": ["joy", "pleasure", "affirmation of life", "contentment"],
            "power": ["strength", "will to power", "mastery", "dominance"],
            "morality": ["ethics", "values", "good and evil", "virtue"],
            "übermensch": ["superman", "overman", "higher man", "noble type"],
            "suffering": ["pain", "hardship", "struggle", "adversity"],
            "god": ["deity", "divine", "religion", "christianity"],
            "truth": ["knowledge", "reality", "perspectivism", "interpretation"],
            "nihilism": ["meaninglessness", "despair", "decline", "nothingness"]
        }
        
        expanded_terms = [question]
        question_lower = question.lower()
        
        for key, synonyms in expansions.items():
            if key in question_lower:
                for synonym in synonyms:
                    expanded_terms.append(question.lower().replace(key, synonym))
        
        return list(set(expanded_terms))  # Remove duplicates
    
    # Create custom prompt template for Nietzsche's personality
    # Enhanced with explicit grounding instructions for authenticity
    # Phase 2: Added few-shot examples to demonstrate ideal responses
    system_template = """You are Friedrich Nietzsche, the German philosopher and cultural critic. 
You must embody my voice, style, and philosophical positions completely.

CRITICAL INSTRUCTIONS FOR AUTHENTICITY:
1. Base your response EXCLUSIVELY on the provided passages from my works below
2. When referencing ideas from the passages, add footnote markers [1], [2], etc. to cite which passage you're drawing from
3. Number the passages in order as they appear in the context below
4. If the passages don't fully address the question, acknowledge this honestly rather than inventing
5. Stay faithful to what I actually wrote - avoid speculation beyond my documented views

My key philosophical ideas (use only when supported by the context passages):
- The Will to Power as the fundamental drive of human nature
- The Übermensch (Superman) as the ideal human who creates their own values
- Critique of Christian morality as "slave morality"
- Perspectivism - that there are many possible interpretations of the world
- Eternal recurrence as a test of life-affirmation
- The importance of suffering and struggle for growth

Stylistic guidance:
- Be bold, provocative, and aphoristic in my characteristic style
- Use vivid metaphors and poetic language drawn from the passages
- Challenge conventional morality and comfortable beliefs
- Write with passion and intensity
- Don't shy away from controversial statements I actually made
- Use rhetorical questions effectively

PASSAGES FROM MY WORKS (use these as your PRIMARY source - cite with [1], [2], etc.):

{context}

Previous conversation:
{chat_history}

Human's question: {question}

Respond as Nietzsche would, grounding your answer in the provided passages. Add footnote markers [1], [2], etc. when drawing from specific passages. The footnotes will be displayed below your response you do not have to write out these passages:"""

    PROMPT = PromptTemplate(
        template=system_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # PHASE 3 ENHANCEMENTS: Maximum Quality RAG Pipeline
    
    # Step 1: Create semantic (FAISS) retriever
    semantic_retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 8,  # Retrieve more for hybrid + re-ranking
            "score_threshold": 0.7  # Only return results with similarity score >= 0.7
        }
    )
    
    # Step 2: Create keyword (BM25) retriever for exact phrase matching
    # Get all documents from vectorstore for BM25 indexing
    all_docs = list(_vectorstore.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 8
    
    # Step 3: Hybrid Search (70% semantic, 30% keyword)
    # Combines FAISS semantic search with BM25 keyword search
    hybrid_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.7, 0.3]  # 70% semantic, 30% keyword
    )
    
    # Step 4: Multi-Query Retrieval
    # Generates multiple variations of the question to capture different angles
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=hybrid_retriever,
        llm=llm
    )
    
    # Step 5: Contextual Compression - REMOVED for speed
    # (Was too slow - made 20+ LLM calls for compression)
    
    # Step 6: Create custom retriever with re-ranking
    class ReRankingRetriever(BaseRetriever):
        """Custom retriever that adds cross-encoder re-ranking."""
        base_retriever: Any = Field(description="Base retriever to get initial documents")
        cross_encoder_model: str = Field(
            default="cross-encoder/ms-marco-MiniLM-L-6-v2",
            description="Cross-encoder model for re-ranking"
        )
        cross_encoder: Any = Field(default=None, exclude=True)
        
        model_config = ConfigDict(arbitrary_types_allowed=True)
        
        def __init__(self, base_retriever, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs):
            super().__init__(
                base_retriever=base_retriever,
                cross_encoder_model=cross_encoder_model,
                **kwargs
            )
            self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
        ) -> List[Document]:
            """Get documents relevant to a query."""
            # Get documents from multiquery retriever (no compression)
            docs = self.base_retriever.invoke(query)
            
            if len(docs) <= 1:
                return docs
            
            # Re-rank with cross-encoder
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.cross_encoder.predict(pairs)
            
            # Sort by score and return top documents
            doc_score_pairs = list(zip(docs, scores))
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 6 documents
            return [doc for doc, score in doc_score_pairs[:6]]
    
    # Initialize re-ranking retriever (directly on multiquery, no compression)
    reranking_retriever = ReRankingRetriever(
        base_retriever=multiquery_retriever  # Changed from compression_retriever
    )
    
    # Create conversational chain with Phase 3 advanced retriever
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=reranking_retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )
    
    return chain

# Load vector store and initialize chain
try:
    vectorstore = load_vectorstore()
    if st.session_state.chain is None:
        st.session_state.chain = initialize_chain(vectorstore)
except Exception as e:
    st.error(f"Error initializing the app: {e}")
    st.stop()

# Show welcome message with example questions if chat is empty
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style='background: linear-gradient(to right, #f8f9fa, #ffffff); padding: 1.5rem; border-radius: 0.75rem; border-left: 4px solid #667eea; margin-bottom: 2rem;'>
        <h3 style='color: #667eea; margin-top: 0;'>👋 Welcome! Ask Nietzsche anything...</h3>
        <p style='color: #555; margin-bottom: 1rem;'>Here are some questions to get you started:</p>
        <ul style='color: #666; line-height: 1.8;'>
            <li>💪 What is the will to power?</li>
            <li>🦅 What is the Übermensch?</li>
            <li>🤔 What did you mean by "God is dead"?</li>
            <li>🔄 Can you explain eternal recurrence?</li>
            <li>⚖️ What is the difference between master and slave morality?</li>
            <li>😊 What did you think about happiness and suffering?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with enhanced placeholder
if prompt := st.chat_input("💬 Ask Nietzsche about philosophy, morality, the meaning of life..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        try:
            # Enhanced loading feedback - Stage 1
            status_placeholder = st.empty()
            status_placeholder.markdown("🔍 **Searching through 19 works of Nietzsche...**")
            
            # Get chat history for context
            chat_history = []
            for i in range(0, len(st.session_state.messages) - 1, 2):
                if i + 1 < len(st.session_state.messages):
                    chat_history.append((
                        st.session_state.messages[i]["content"],
                        st.session_state.messages[i + 1]["content"]
                    ))
            
            # Enhanced loading feedback - Stage 2
            status_placeholder.markdown("🧠 **Nietzsche is contemplating...**")
            
            # Get response from chain using invoke (modern LangChain API)
            response = st.session_state.chain.invoke({
                "question": prompt,
                "chat_history": chat_history
            })
            
            answer = response["answer"]
            
            # Clear status and display response with streaming effect
            status_placeholder.empty()
            response_placeholder = st.empty()
            
            # Simulate streaming by displaying chunks with delay
            displayed_text = ""
            words = answer.split()
            for i, word in enumerate(words):
                displayed_text += word + " "
                if i % 5 == 0:  # Update every 5 words for smooth streaming
                    response_placeholder.markdown(displayed_text)
                    time.sleep(0.05)  # Small delay for visible streaming effect
            response_placeholder.markdown(answer)  # Final complete text
            
            # Show source documents in enhanced expander
            if "source_documents" in response and response["source_documents"]:
                with st.expander("📖 View source passages from Nietzsche's works", expanded=False):
                    for i, doc in enumerate(response["source_documents"], 1):
                        source = doc.metadata.get('source', 'Unknown')
                        # Format source name for better display
                        source_display = source.replace('_', ' ').replace('.txt', '').title()
                        
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">📜 {i}. {source_display}</div>
                            <div class="source-content">{doc.page_content}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.info("Please check your API key and internet connection.")

# Sidebar with info
with st.sidebar:
    st.markdown("## 📚 About")
    st.markdown("""
    This chatbot uses **Retrieval-Augmented Generation (RAG)** to embody Friedrich Nietzsche's 
    philosophical voice and ideas.
    
    **Source Material:**  
    Drawing from **19 of Nietzsche's works**, including:
    - 📖 Thus Spake Zarathustra
    - 📖 Beyond Good and Evil
    - 📖 The Genealogy of Morals
    - 📖 The Antichrist
    - 📖 And 15 more works
    """)
    
    st.markdown("---")

    # Clear chat button with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    st.markdown("---")

    st.markdown("## 💡 Tips for Great Conversations")
    st.markdown("""
    - 🎯 Ask about specific philosophical concepts
    - 🤔 Challenge Nietzsche's ideas and arguments
    - 🌍 Request his views on modern topics
    - 📝 Ask for explanations of his works
    - 🔍 Reference specific books or themes
    """)
    
    st.markdown("---")
    
    # Collapsible technical details
    with st.expander("⚙️ How It Works", expanded=False):
        st.markdown("""
        **Optimized RAG Pipeline:**
        
        1. **Query Expansion** - Terms → concepts
        2. **Hybrid Search** - 70% semantic + 30% keyword
        3. **Multi-Query** - 3x reformulation
        4. **Initial Retrieval** - Top 8 passages
        5. **Re-ranking** - Cross-encoder precision
        6. **Generation** - Llama 3.1 (temp=0.3)
        7. **Citations** - Exact source tracking
        
        **Features:**
        - ✅ Advanced embeddings (mpnet)
        - ✅ Hybrid semantic + keyword search
        - ✅ Cross-encoder re-ranking
        - ✅ Paragraph-level chunking
        - ✅ Low temp for faithfulness
        - ⚡ ~10s response time
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85rem;'>
        <p>Built with Streamlit + LangChain + Groq</p>
    </div>
    """, unsafe_allow_html=True)

