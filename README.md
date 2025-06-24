**fyndo** is an enterprise search platform designed to provide intelligent document discovery and knowledge retrieval across organizational data silos. It is powered by AI-driven tag generation and an advanced search module for precise, context-aware results.

## Modules

### 1. Tag Generation Agent

**Location:** `agents/tag_generator.py`

**Responsibilities:**  
- Automate extraction of source and semantic content tags from documents (supports S3, PDFs, etc.).
- Use an LLM (e.g., Gemini) to generate and score content-relevant tags and produce concise file descriptions.
- Store all tags, scores, and descriptions in MongoDB for fast retrieval.
- Support robust batch workflows, resumable processing, and error handling.

### 2. Intelligent Search Agent

**Location:** `streamlit_app.py`

**Responsibilities:**  
- Provide a modern, user-friendly search experience via Streamlit.
- Allow filtering by source tags (e.g., “github”, “notion”, etc.) and/or context-aware AI-generated content tags.
- Use an LLM to match user queries to content tags and calculate document relevance.
- Return ranked, transparent results with tag and relevance breakdowns.

---

## Data Flow

1. **Tag Generation**
    - Extract source tags from file path/type.
    - Parse document content (e.g., PDF text extraction).
    - Use an LLM to generate content tags + a summary description.
    - Persist all metadata to MongoDB.

2. **Search**
    - User submits a query and optionally selects source tags.
    - Retrieve relevant docs from MongoDB using these tags.
    - LLM further matches and scores content tags against the user query.
    - Return and display the top-ranked results.

---

## Technologies Used

- **Python** (core backend)
- **Streamlit** (UI)
- **MongoDB** (metadata and tag storage)
- **Google Gemini LLM** (tag generation and search intelligence)
- **S3** or similar (document storage)
- **pdfplumber** (for PDF parsing)
- **StateGraph** (workflow orchestration for batch tagging)
## References

- [Tag Generation Agent (`agents/tag_generator.py`)](https://github.com/AI-Mercenary/fyndo/blob/main/agents/tag_generator.py)
- [Intelligent Search Agent (`streamlit_app.py`)](https://github.com/AI-Mercenary/fyndo/blob/main/streamlit_app.py)
