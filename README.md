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

### Step 3: Visualize Embeddings (Optional)
1. Click **"Visualize Semantic Clusters"** in the sidebar
2. See a PCA visualization of your document embeddings
3. Explore how documents are semantically grouped

---

## 🔧 Troubleshooting

### Issue: "GOOGLE_API_KEY not found"
**Solution:** Ensure your `.env` file is in the project root with your API key
```bash
# Verify the file exists
cat .env
# Output should show: GOOGLE_API_KEY="sk-..."
```

### Issue: "No documents found in data/"
**Solution:** Add documents to the data directory
```bash
# Check what's in the directory
ls -la data/

# Add files
cp ~/Documents/my-resume.pdf data/
cp ~/Documents/cover-letter.txt data/
```

### Issue: "ModuleNotFoundError" when running
**Solution:** Activate the virtual environment
```bash
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate  # Windows
```

### Issue: Port 8501 already in use
**Solution:** Use a different port
```bash
streamlit run app.py --server.port 8502
```

### Issue: Slow performance on first run
**Solution:** This is normal. First-time embedding takes longer. Subsequent runs use cached ChromaDB data.

---

## 📁 Project Structure

```
rag-system/
├── app.py                 # Main Streamlit entry point
├── requirements.txt       # Python dependencies
├── .env                   # API keys (create this file)
├── .venv/                 # Virtual environment (auto-created)
├── data/                  # Your documents go here
│   └── *.txt, *.pdf       # Supported file formats
├── database/              # ChromaDB persistent storage
│   └── chroma.sqlite3
├── src/                   # Core RAG logic
│   ├── ingestion.py       # Document processing
│   ├── retrieval.py       # Hybrid search + re-ranking
│   ├── generation.py      # Gemini response generation
│   ├── visualization.py   # PCA clustering visualization
│   └── metrics.py         # Evaluation metrics
├── PLAN.md                # Development roadmap
└── DESIGN.md              # System architecture
```

---

## 🚀 Quick Start (Copy-Paste)

```bash
# 1. Navigate to project
cd /Users/siddhartha/Documents/Python/rag-system

# 2. Activate environment
source .venv/bin/activate

# 3. Create .env file with your API key
cat > .env << EOF
GOOGLE_API_KEY="your-google-api-key-here"
EOF

# 4. Add your documents
mkdir -p data
cp ~/Documents/my-resume.pdf data/

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` automatically.

---

## 🔑 API Key Setup (Detailed)

### Getting Your Google API Key
1. Open [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"** button
3. Select your project (or create new one)
4. Click **"Create API Key in new project"**
5. Copy the key displayed

### Adding to Your Project
Create a `.env` file in the project root:
```bash
# macOS/Linux
echo 'GOOGLE_API_KEY="paste-your-key-here"' > .env

# Windows (PowerShell)
@'
GOOGLE_API_KEY="paste-your-key-here"
'@ | Out-File .env -Encoding UTF8
```

### Verify It Works
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key loaded!' if os.getenv('GOOGLE_API_KEY') else 'Failed!')"
```

---

## 📝 Detailed Usage Walkthrough

### Complete Workflow

**1. Initial Setup (First Time)**
```bash
# Terminal 1: Start the app
cd /Users/siddhartha/Documents/Python/rag-system
source .venv/bin/activate
streamlit run app.py
```

**2. In the Browser (http://localhost:8501)**
- You'll see the RAG chat interface
- Left sidebar has two buttons: **"Build/Update Knowledge Base"** and **"Visualize Semantic Clusters"**

**3. Build Your Knowledge Base**
- Make sure documents are in `./data/` directory
- Click **"Build/Update Knowledge Base"** button
- Wait for "Database Ready!" message
- This step:
  - Reads all `.txt` and `.pdf` files from `./data/`
  - Chunks documents into optimal sizes
  - Generates embeddings (calls Google API)
  - Stores in ChromaDB (persistent across sessions)

**4. Ask Questions**
- In the chat box, type: *"What are my key skills?"*
- System performs:
  - **Retrieval:** Hybrid search (BM25 keyword + vector semantic)
  - **Re-ranking:** FlashrankRerank sorts by relevance
  - **Generation:** Gemini creates response using retrieved context
  - **Display:** Response appears in chat

**5. Explore Embeddings (Optional)**
- Click **"Visualize Semantic Clusters"**
- See PCA 2D projection of your documents
- Helps understand document relationships

---

## ⚙️ Configuration & Customization

### Environment Variables
Edit `.env` to add/modify:
```bash
# Required
GOOGLE_API_KEY="sk-..."

# Optional (defaults used if not set)
EMBEDDING_MODEL="models/text-embedding-004"
GENERATION_MODEL="gemini-1.5-flash"
CHUNK_SIZE=500
CHUNK_OVERLAP=100
```

### Document Processing Options
In `src/ingestion.py`, you can modify:
- **Chunk size:** Larger chunks = more context, slower retrieval
- **Chunk overlap:** Prevents cutting info across chunks
- **File types:** Add support for `.docx`, `.md`, etc.

### Retrieval Settings
In `src/retrieval.py`:
- **BM25 weight:** Adjust keyword search sensitivity
- **Vector weight:** Adjust semantic search sensitivity
- **Re-ranking model:** Switch flashrank strategy

---

## ✅ Verification Checklist

Before sharing or deploying:

- [ ] `.env` file created with valid API key
- [ ] `data/` directory has at least one document
- [ ] Ran `streamlit run app.py` successfully
- [ ] Clicked "Build/Update Knowledge Base" and got "Database Ready!"
- [ ] Asked a test question and got a response
- [ ] No error messages in terminal

---

## 🔍 Debugging & Logs

### View Detailed Logs
```bash
# Run with verbose output
streamlit run app.py --logger.level=debug
```

### Check What's Happening
- **Terminal shows:** API calls, embeddings progress
- **Browser shows:** Chat history, status messages
- **database/ folder:** ChromaDB files (persistent storage)

### Verify Data Ingestion
```bash
# Check documents are being read
ls -la data/

# Check database was created
ls -la database/
```

---

## 🚀 Next Steps

1. **Add More Documents:** Copy PDFs/TXTs to `./data/`
2. **Rebuild Database:** Click "Build/Update" again
3. **Explore Results:** Check visualization and ask targeted questions
4. **Customize:** Edit prompts in `src/generation.py`

---

## 📚 System Architecture

```
User Input (Chat)
    ↓
Retrieval Pipeline (src/retrieval.py)
    ├→ Keyword Search (BM25)
    ├→ Vector Search (Embeddings)
    └→ Re-ranking (FlashrankRerank)
    ↓
Retrieved Context
    ↓
Generation (src/generation.py)
    ├→ Prompt Engineering
    └→ Gemini API Call
    ↓
Response Display (Chat)
```

---

## 💡 Example Questions

Try these queries after building your knowledge base:
- "What is the main topic of these documents?"
- "Summarize the key findings"
- "How does X relate to Y in the documents?"
- "What are the most important points?"
- "Create a timeline of key events"

---

## 🔒 Security Best Practices

⚠️ **CRITICAL:** API keys in `.env` are for development only!

**Production Deployment Checklist:**
- [ ] Use environment secrets (AWS Secrets Manager, Google Secret Manager)
- [ ] Never commit `.env` to version control (verify `.gitignore`)
- [ ] Rotate API keys regularly
- [ ] Implement API rate limiting
- [ ] Add authentication/authorization layer
- [ ] Monitor API usage and costs via Google Cloud Console
- [ ] Use HTTPS for any web deployment
- [ ] Implement input sanitization
- [ ] Add audit logging

---

## 📊 Performance Optimization

### For Speed
- Reduce chunk size: 300-500 tokens
- Lower `top_k` retrieval: 3-5 results
- Use smaller embedding models
- Enable caching for frequent queries

### For Quality
- Increase chunk size: 1000-1500 tokens
- Increase overlap: 150-200 tokens
- Higher `top_k`: 8-10 results
- Use stronger models (gemini-pro instead of flash)

### Cost Control
- Monitor API usage: `Google Cloud Console → Billing`
- Set up usage alerts
- Cache results for identical queries
- Batch process during off-peak hours

---

## 📚 Documentation & Resources

- **Full Architecture:** See [DESIGN.md](DESIGN.md)
- **Source Code:** See individual files in [src/](src/)
- **External References:**
  - [LangChain Documentation](https://docs.langchain.com)
  - [ChromaDB Vector Store](https://docs.trychroma.com)
  - [Google Generative AI API](https://ai.google.dev)
  - [Streamlit Framework](https://docs.streamlit.io)

---

## ❓ FAQ

**Q: Can I use different LLMs (OpenAI, Claude, etc.)?**
A: Yes! Edit [src/generation.py](src/generation.py) to swap LLM providers. LangChain supports most major providers.

**Q: How do I add more documents?**
A: Copy files to `data/` directory and click "Build/Update Knowledge Base" again. Changes are automatic.

**Q: What documents formats are supported?**
A: Currently `.txt` and `.pdf`. To add `.docx`, `.md`, etc., update [src/ingestion.py](src/ingestion.py).

**Q: Is this production-ready?**
A: The architecture is solid, but add security monitoring, rate limiting, and error handling for production.

**Q: How much does it cost?**
A: Google Gemini API has free tier (up to 60 requests/min). Monitor costs in [Google Cloud Console](https://console.cloud.google.com).

**Q: Can I deploy this online?**
A: Yes! Streamlit has free [Cloud](https://streamlit.io/cloud) hosting, or use Docker/Kubernetes for enterprise deployment.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Submit a pull request

See [DESIGN.md](DESIGN.md) for the development roadmap and planned features.

---

## 📞 Support & Troubleshooting

**Having Issues?** Try this order:
1. Check [Troubleshooting section](#-troubleshooting) above
2. Verify `data/` has documents: `ls -la data/`
3. Check `.env` file exists: `cat .env | grep GOOGLE_API_KEY`
4. Run with debug logging: `streamlit run app.py --logger.level=debug`
5. Review [DESIGN.md](DESIGN.md) for architecture insights

---

**Happy RAGing! 🚀**
