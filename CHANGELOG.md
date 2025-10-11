# Changelog

All notable changes to the Nietzsche Chatbot project.

---

## [Phase 2] - Advanced RAG Implementation

### Added
- **Multi-Query Retrieval**: Questions are automatically reformulated 3-5 ways for broader passage coverage
- **Contextual Compression**: LLM extracts only relevant portions from retrieved documents, filtering noise
- **Enhanced Retrieval Pipeline**: Base retrieval increased from k=6 to k=10 before compression
- **Phase 2 Documentation**: Created PHASE2_IMPLEMENTATION.md with complete technical details

### Changed
- Updated `app.py` imports to include MultiQueryRetriever, ContextualCompressionRetriever, LLMChainExtractor
- Modified initialize_chain() function to use advanced retrieval pipeline
- Updated README.md to reflect Phase 2 features and pipeline
- Updated sidebar "How it works" section with Phase 2 architecture

### Performance
- Response time increased from ~2-3s to ~4-6s (2-3x slower)
- API calls per question increased from ~1-2 to ~6-9 calls
- Significantly improved response quality and authenticity

---

## [Phase 1] - Core Improvements

### Added
- **Paragraph-Based Chunking**: Preserves complete thoughts (1200 char chunks vs fixed 1000)
- **Enhanced System Prompt**: Explicit grounding instructions with 5 authenticity rules
- **Phase 1 Documentation**: Created PHASE1_IMPROVEMENTS.md

### Changed
- Reduced LLM temperature from 0.7 → 0.3 for more faithful responses
- Increased retrieval from k=4 → k=6 documents
- Updated text splitter to prioritize paragraphs and sentence boundaries
- Added chunk statistics tracking (avg, min, max sizes)
- Updated README with Phase 1 enhancements

---

## [Original Build] - Base Implementation

### Added
- Streamlit chat interface with st.chat_message and st.chat_input
- FAISS vector store with HuggingFace embeddings (all-MiniLM-L6-v2)
- Groq API integration (Llama 3.1 8B Instant)
- ConversationalRetrievalChain for context-aware responses
- Source document citations in expandable UI
- 19 Nietzsche works loaded from preprocessed texts
- Personality prompt capturing Nietzsche's philosophical style
- Secrets management for API keys
- README, SETUP_GUIDE, and documentation

### Configuration
- Temperature: 0.7
- Max tokens: 1024
- Initial retrieval: k=4
- Chunk size: 1000 chars, overlap: 200

---

## Future

### [Phase 3] - Planned
- Better embeddings (all-mpnet-base-v2)
- Hybrid search (semantic + keyword BM25)
- Cross-encoder re-ranking
- Query expansion with related terms

---

**Current Version**: Phase 2 Complete ✅  
**Last Updated**: Implementation of advanced RAG features  
**Status**: Production Ready

