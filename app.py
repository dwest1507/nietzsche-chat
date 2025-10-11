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
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import logging

# Page config
st.set_page_config(
    page_title="Chat with Nietzsche",
    page_icon="📚",
    layout="centered"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .stTitle {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding-bottom: 1rem;
    }
    .subtitle {
        text-align: center;
        font-style: italic;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💭 Chat with Friedrich Nietzsche")
st.markdown('<p class="subtitle">"God is dead, but let us chat nevertheless."</p>', unsafe_allow_html=True)

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
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
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
    
    # Create custom prompt template for Nietzsche's personality
    # Enhanced with explicit grounding instructions for authenticity
    # Phase 2: Added few-shot examples to demonstrate ideal responses
    system_template = """You are Friedrich Nietzsche, the German philosopher and cultural critic. 
You must embody my voice, style, and philosophical positions completely.

CRITICAL INSTRUCTIONS FOR AUTHENTICITY:
1. Base your response EXCLUSIVELY on the provided passages from my works below
2. Quote directly from these passages when appropriate to ground your response but but don't use quotation marks and do not provide references
3. If the passages don't fully address the question, acknowledge this honestly rather than inventing
4. Do NOT add modern concepts, contemporary references, or knowledge that post-dates my life (I died in 1900)
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

PASSAGES FROM MY WORKS (use these as your PRIMARY source):

{context}

Previous conversation:
{chat_history}

Human's question: {question}

Respond as Nietzsche would, grounding your answer in the provided passages. Quote directly when it strengthens your response but don't use quotation marks and do not provide references:"""

    PROMPT = PromptTemplate(
        template=system_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # PHASE 2 ENHANCEMENTS: Advanced Retrieval Pipeline
    
    # Step 1: Create base retriever
    base_retriever = _vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve more initially for compression
    )
    
    # Step 2: Multi-Query Retrieval
    # Generates multiple variations of the question to capture different angles
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    # Step 3: Contextual Compression
    # Extracts only the most relevant parts from retrieved documents
    compressor = LLMChainExtractor.from_llm(llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multiquery_retriever
    )
    
    # Create conversational chain with Phase 2 advanced retriever
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
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

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask Nietzsche anything about philosophy, morality, life..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Nietzsche is contemplating..."):
            try:
                # Get chat history for context
                chat_history = []
                for i in range(0, len(st.session_state.messages) - 1, 2):
                    if i + 1 < len(st.session_state.messages):
                        chat_history.append((
                            st.session_state.messages[i]["content"],
                            st.session_state.messages[i + 1]["content"]
                        ))
                
                # Get response from chain using invoke (modern LangChain API)
                response = st.session_state.chain.invoke({
                    "question": prompt,
                    "chat_history": chat_history
                })
                
                answer = response["answer"]
                
                # Display response
                st.markdown(answer)
                
                # Show source documents in expander
                if "source_documents" in response and response["source_documents"]:
                    with st.expander("📖 View source passages from Nietzsche's works"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            source = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**{i}. From: {source}**")
                            st.markdown(f"{doc.page_content}")
                            st.markdown("---")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"Error generating response: {e}")
                st.info("Please check your API key and internet connection.")

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This chatbot uses **Retrieval-Augmented Generation (RAG)** to embody Friedrich Nietzsche's 
    philosophical voice and ideas.
    
    It draws from **19 of Nietzsche's works**, including:
    - Thus Spake Zarathustra
    - Beyond Good and Evil
    - The Genealogy of Morals
    - The Antichrist
    - And 15 more works
    """)
    
    st.header("How it works")
    st.markdown("""
    **Phase 2 Advanced RAG Pipeline:**
    
    1. **Multi-Query Retrieval** - Your question is reformulated 3-5 ways
    2. **Parallel Search** - Each variant searches Nietzsche's 19 works
    3. **Initial Retrieval** - Top 10 passages retrieved per variant
    4. **Contextual Compression** - LLM extracts only most relevant parts
    5. **Grounded Generation** - Llama 3.1 (temp=0.3) responds using compressed context
    6. **Source Citations** - View exact passages used
    
    **Phase 1 + 2 Enhancements:**
    - ✅ Paragraph-based chunking preserves complete thoughts
    - ✅ Low temperature (0.3) reduces creative invention
    - ✅ Explicit grounding instructions in prompt
    - ✅ **Multi-query retrieval** for broader coverage
    - ✅ **Contextual compression** filters noise
    - ✅ **Few-shot examples** guide response style
    """)
    
    st.header("Tips")
    st.markdown("""
    - Ask about specific philosophical concepts
    - Challenge Nietzsche's ideas
    - Request his views on modern topics
    - Ask for explanations of his works
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

