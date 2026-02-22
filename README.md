# 🚀 Gemini Production RAG

A production-ready Retrieval-Augmented Generation (RAG) application powered by Google Gemini AI. Ask questions about your documents and get intelligent, context-aware responses.

**Key Features:**
- 🔍 **Hybrid Search** - Combines keyword (BM25) and semantic (vector) retrieval
- ⭐ **Intelligent Re-ranking** - FlashrankRerank ensures most relevant results
- 🤖 **Gemini Integration** - Leverages Google's advanced LLMs
- 📊 **Visualization** - Explore document semantic clusters with PCA
- 💬 **Conversational UI** - Chat interface with message history
- 📁 **Multi-format Support** - Process TXT and PDF documents

---

## 📋 Prerequisites

- **Python 3.10+**
- **Google API Key** (for Gemini and embeddings)
- **macOS/Linux/Windows** with terminal access

---

## 🛠️ Installation

### 1. Clone & Setup Project
```bash
# Navigate to project directory
cd /Users/siddhartha/Documents/Python/RAG

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure API Keys
Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY="your-google-api-key-here"
```

**Get your Google API Key:**
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and paste into `.env`

### 4. Prepare Your Documents
```bash
# Create data directory if it doesn't exist
mkdir -p data

# Add your documents (TXT or PDF files)
cp /path/to/documents/* data/
```

Supported formats:
- `.txt` - Plain text files
- `.pdf` - PDF documents

---

## 🚀 Running the Application

### Start Streamlit App
```bash
streamlit run app.py
```

This will:
- Open `http://localhost:8501` in your browser
- Display the RAG chat interface
- Keep the app running until you stop it (Ctrl+C)

---

## 📖 How to Use

### Step 1: Build Knowledge Base
1. Click **"Build/Update Knowledge Base"** in the left sidebar
2. Wait for the spinner to complete - documents are:
   - Loaded from `./data/`
   - Split into optimized chunks
   - Embedded using Google's text-embedding-004 model
   - Stored in ChromaDB (persistent)

**Status:** You'll see "Database Ready!" when complete

### Step 2: Ask Questions
1. Type your question in the chat input box
2. The system will:
   - Retrieve relevant document chunks (hybrid search)
   - Re-rank results for accuracy
   - Generate response using Gemini
3. View the response in the chat history

**Example Questions:**
- "What is the main topic of these documents?"
- "Summarize the key findings"
- "How does X relate to Y in the documents?"

### Step 3: Visualize Clusters (Optional)
1. Click **"Visualize Semantic Clusters"** in the sidebar
2. A PCA-reduced plot shows how documents are semantically related
3. Blue dots represent document chunks in 2D space

---

## 📁 Project Structure

```
RAG/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # API keys (DO NOT COMMIT)
├── .gitignore            # Git ignore rules
├── README.md             # This file
├── DESIGN.md             # Architecture & future improvements
│
├── src/                  # Source modules
│   ├── __init__.py
│   ├── ingestion.py      # Document loading & chunking
│   ├── retrieval.py      # Hybrid search & re-ranking
│   ├── generation.py     # RAG chain & LLM integration
│   ├── visualization.py  # Embedding visualization
│   ├── metrics.py        # RAG evaluation (future)
│   └── __pycache__/
│
├── data/                 # Your documents (TXT/PDF)
│   └── (add your files here)
│
└── database/             # Vector store (auto-created)
    ├── chroma.sqlite3
    └── [vector IDs]/
```

---

## 🔧 Configuration

### Document Chunking
Edit `src/ingestion.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Adjust chunk size
    chunk_overlap=100       # Adjust overlap
)
```

### Retrieval Weights
Edit `src/retrieval.py`:
```python
ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever],
    weights=[0.4, 0.6]      # BM25 vs Vector weights
)
```

### LLM Parameters
Edit `src/generation.py`:
```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,           # 0=deterministic, 1=creative
    max_output_tokens=2048
)
```

---

## 🐛 Common Issues & Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'langchain'`
**Solution:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: `No documents found in data/`
**Solution:**
- Create the `data/` folder: `mkdir -p data`
- Add TXT or PDF files to `data/`
- Click "Build/Update Knowledge Base" again

### Issue: `GOOGLE_API_KEY not found`
**Solution:**
- Create `.env` file in project root
- Add: `GOOGLE_API_KEY="your-key-here"`
- Ensure it's not in `.gitignore` (for development)

### Issue: `Streamlit connection refused`
**Solution:**
```bash
# Kill any existing process on port 8501
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run app.py --server.port 8502
```

### Issue: Slow embedding/generation
**Solution:**
- Reduce chunk size in `ingestion.py`
- Reduce `max_output_tokens` in `generation.py`
- Check your internet connection (API calls are network-dependent)

---

## 📊 Performance Tips

1. **Faster Processing:**
   - Use smaller chunk sizes (500-800 tokens)
   - Reduce `top_k` values in retrieval

2. **Better Results:**
   - Increase chunk size (1200-1500 tokens)
   - Increase overlap (150-200)
   - Use more retrieval results (k=10)

3. **Cost Optimization:**
   - Cache frequently asked questions
   - Batch process documents during off-peak hours
   - Monitor API usage via Google Cloud Console

---

## 🔒 Security Best Practices

⚠️ **CRITICAL:** API keys in `.env` are for development only!

**Production Deployment:**
- [ ] Use environment secrets (AWS Secrets Manager, Google Secret Manager)
- [ ] Never commit `.env` to version control
- [ ] Rotate API keys regularly
- [ ] Implement API rate limiting
- [ ] Add authentication/authorization layer
- [ ] Monitor API usage and costs

See [DESIGN.md](DESIGN.md) for security roadmap.

---

## 📈 Monitoring & Debugging

### Enable Logging
Add this to `app.py` for debugging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check API Quotas
- Visit [Google Cloud Console](https://console.cloud.google.com)
- Monitor API usage and quotas
- Set up billing alerts

### Performance Profiling
```bash
# Profile memory usage
pip install memory-profiler
python -m memory_profiler src/ingestion.py
```

---

## 🎯 Next Steps

1. **Read [DESIGN.md](DESIGN.md)** for:
   - Complete architecture overview
   - Technology stack details
   - Future improvement roadmap
   - Performance characteristics

2. **Customize for Your Use Case:**
   - Add domain-specific prompts
   - Adjust chunking strategy
   - Integrate additional data sources

3. **Evaluate RAG Quality:**
   - Use RAGAS framework (installed)
   - Implement custom metrics
   - Track query-response quality

---

## 📚 Documentation

- **Architecture:** See [DESIGN.md](DESIGN.md)
- **Source Code:** See individual files in `src/`
- **External Links:**
  - [LangChain Docs](https://docs.langchain.com)
  - [ChromaDB Docs](https://docs.trychroma.com)
  - [Google AI API](https://ai.google.dev)
  - [Streamlit Docs](https://docs.streamlit.io)

---

## 🧪 Development

### Code Structure
- Clean modular design with separated concerns
- Each module has clear responsibilities
- Easy to extend with custom components

### Adding New Retrievers
Edit `src/retrieval.py`:
```python
# Add new retriever to ensemble
new_retriever = MyCustomRetriever.from_documents(chunks)
ensemble = EnsembleRetriever(
    retrievers=[bm25, vector_retriever, new_retriever],
    weights=[0.3, 0.4, 0.3]
)
```

### Testing
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/
```

---

## 📝 License

This project is open source. Check LICENSE file for details.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

See [DESIGN.md](DESIGN.md) for development roadmap.

---

## ❓ FAQ

**Q: Can I use different LLMs?**
A: Yes! Edit `src/generation.py` to use Claude, OpenAI, or other providers.

**Q: How do I add more documents?**
A: Drop TXT/PDF files in `data/` and click "Build/Update Knowledge Base".

**Q: Is this production-ready?**
A: The architecture is production-ready, but add security/monitoring for production deployment.

**Q: How much does it cost?**
A: Google Gemini has free tier limits. Monitor via Google Cloud Console.

---

## 📞 Support

For issues:
1. Check **Troubleshooting** section above
2. Review [DESIGN.md](DESIGN.md) for architecture
3. Check console output for error messages
4. Enable debug logging for more details

---

**Happy RAGing! 🚀**
