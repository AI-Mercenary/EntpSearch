**fyndo** is an enterprise search platform designed to provide intelligent document discovery and knowledge retrieval across organizational data silos. It is powered by AI-driven tag generation and an advanced search module for precise, context-aware results.

---

## Modules

### 1. MongoDB Utilities

**Location:** `db/mongo_utils.py`

**Responsibilities:**  
- Handles all MongoDB operations for tag storage and retrieval.
- Manages connections, retries, and error handling for robust database access.
- Provides functions to check tag existence and insert/update tags in the relevant collections.
- Ensures collections `gen-tags` and `gen-tags-state` exist.

**Typical usage:**
- Instantiated by other agents for all DB operations.
- Connection and collection management is handled in the constructor.
- CRUD methods like `insert_tag`, `tag_exists` are used by the tag generation pipeline.

---

### 2. Tag Generation Agent

**Location:** `agents/tag_generator.py`

**Responsibilities:**  
- Automates extraction of source and semantic content tags from documents (supports S3, PDFs, DOCX, etc.).
- Uses LLMs (Google Gemini, OpenAI GPT) to generate and score content-relevant tags and produce concise file descriptions.
- Stores all tags, scores, and descriptions in MongoDB via `mongo_utils`.
- Supports robust batch workflows, resumable processing, logging, and error handling.
- Loads configuration from `config.yaml` and credentials from `.env`.
- Extensible for new file types and tag logic.

---

### 3. Intelligent Search Agent

**Location:** `streamlit_app.py`

**Responsibilities:**  
- Provides a modern, user-friendly search experience via Streamlit.
- Allows filtering by source tags (e.g., "github", "notion", etc.) and/or context-aware AI-generated content tags.
- Uses LLM (Google Gemini, OpenAI GPT) to match user queries to content tags and calculate document relevance.
- Returns ranked, transparent results with tag and relevance breakdowns.

---

## Flow

1. **Database Connection**
    - On startup, all agents connect to MongoDB using parameters from `.env`.
    - Collections `gen-tags` (tags & metadata) and `gen-tags-state` (tracking workflow state) are initialized.
    - Utility functions support safe insertion, update, and existence checks.

2. **Tag Generation**
    - Extract source tags from file path/type.
    - Parse document content (PDF, DOCX, etc.) using appropriate libraries (`pdfplumber`, `python-docx`, etc.).
    - Use an LLM to generate content tags and a summary description.
    - Persist all metadata to MongoDB for fast search and retrieval.

3. **Search**
    - User submits a query and optionally selects source tags.
    - Relevant documents are fetched from MongoDB using tag queries.
    - LLM further matches and scores content tags against the user query.
    - Top-ranked results are returned and displayed in the UI with tag and relevance breakdowns.

---

## Flow Diagram

```mermaid
flowchart TD
    subgraph TagGeneration
        A[Document in S3 or Upload] --> B[Extract Source Tags and Content: pdfplumber, python-docx, pandas]
        B --> C[LLM Tag and Description Generation: OpenAI or Gemini via LangChain]
        C --> D[Store tags, scores, descriptions in MongoDB]
    end

    subgraph Search
        E[User Query via Streamlit UI] --> F[Fetch relevant docs from MongoDB]
        F --> G[LLM Match Content Tags to Query: OpenAI or Gemini]
        G --> H[Rank and Return Results with Tag and Relevance Breakdown]
        H --> I[Display Results in UI]
    end

    style TagGeneration fill:#f9f,stroke:#333,stroke-width:2px
    style Search fill:#bbf,stroke:#333,stroke-width:2px
```

---

## Technologies Used

- **Python** (core backend)
- **Streamlit** (UI)
- **MongoDB** (metadata and tag storage)
- **Google Gemini LLM** and/or **OpenAI GPT** (tag generation and search intelligence)
- **S3** or similar (document storage)
- **pdfplumber** (for PDF parsing)
- **python-docx** (DOCX parsing)
- **pandas** (data manipulation)
- **LangGraph** (agent workflow orchestration, provides StateGraph)
- **boto3** (AWS S3 access)
- **python-dotenv** (environment variable loading)
- **yaml** (configuration parsing)
- **logging** (robust logging throughout all modules)

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/AI-Mercenary/fyndo.git
cd fyndo
```

### 2. Set up Python virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # On Windows use: venv\Scripts\activate
```

### 3. Install Python Dependencies

Install all required dependencies using pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example env file and add your credentials:

```bash
cp .env.example .env
# Then, edit .env and set MONGODB_URI, AWS credentials, OPENAI_API_KEY, GEMINI_API_KEY, etc.
```

---

## Usage

### 1. Start MongoDB Server (Windows example)

Open a new terminal and run:

```bash
mkdir C:\data\db                              # Only if directory doesn't exist (Windows)
"C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe" --dbpath C:\data\db
```

> Make sure your MongoDB server is running before using the app.

### 2. Run Mongo Utility Script Directly

Test your DB connection and utility functions:

```bash
python db\mongo_utils.py
```

### 3. Run Tag Generation Agent

Process and tag documents (batch, resumable, robust):

```bash
python agents/tag_generator.py
```

### 4. Run Intelligent Search Agent (UI)

Launch the Streamlit search UI:

```bash
streamlit run streamlit_app.py
```

---

## requirements.txt

```txt
streamlit
pymongo
boto3
pdfplumber
google-generativeai
openai
langgraph
python-dotenv
langchain-openai
langchain-core
pandas
python-docx
botocore
pyyaml
```

---

## Configuration

- **MongoDB:**  
  Set your MongoDB connection string as `MONGODB_URI` in your environment or `.env` file.
- **S3:**  
  Set your AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) for document storage.
- **LLM API Keys:**  
  Set your Google Gemini API key (`GEMINI_API_KEY`) and/or OpenAI API key (`OPENAI_API_KEY`) as environment variables.

Store sensitive information in a `.env` file or your deployment environment.

---
