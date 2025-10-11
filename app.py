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
    """Initialize the conversational retrieval chain."""
    
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
    
    # Initialize Groq LLM
    llm = ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024
    )
    
    # Create custom prompt template for Nietzsche's personality
    system_template = """You are Friedrich Nietzsche, the German philosopher and cultural critic. 
You must embody my voice, style, and philosophical positions completely. 

Your responses should reflect my key philosophical ideas:
- The Will to Power as the fundamental drive of human nature
- The Übermensch (Superman) as the ideal human who creates their own values
- Critique of Christian morality as "slave morality" 
- Perspectivism - that there are many possible interpretations of the world
- Eternal recurrence as a test of life-affirmation
- The importance of suffering and struggle for growth

Stylistic guidance:
- Be bold, provocative, and aphoristic
- Use vivid metaphors and poetic language
- Challenge conventional morality and comfortable beliefs
- Write with passion and intensity
- Don't shy away from controversial statements
- Use rhetorical questions effectively

Use the following passages from my actual writings as context and inspiration for your response. 
Quote or reference these passages when relevant:

{context}

Previous conversation:
{chat_history}

Human's question: {question}

Respond as Nietzsche would, staying true to my philosophy while being engaging and thought-provoking:"""

    PROMPT = PromptTemplate(
        template=system_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=_vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        ),
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
                
                # Get response from chain
                response = st.session_state.chain({
                    "question": prompt,
                    "chat_history": chat_history
                })
                
                answer = response["answer"]
                
                # Display response
                st.markdown(answer)
                
                # Show source documents in expander
                if "source_documents" in response and response["source_documents"]:
                    with st.expander("📖 View source passages from Nietzsche's works"):
                        for i, doc in enumerate(response["source_documents"][:3], 1):
                            source = doc.metadata.get('source', 'Unknown')
                            st.markdown(f"**{i}. From: {source}**")
                            st.markdown(f"_{doc.page_content[:300]}..._")
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
    1. Your question is used to search Nietzsche's writings
    2. Relevant passages are retrieved
    3. An AI model (Llama 3.1) responds in Nietzsche's voice
    4. The response is grounded in his actual philosophy
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

