import streamlit as st
import google.generativeai as genai
from pymongo import MongoClient
import json
import re

# Configure Streamlit page
st.set_page_config(
    page_title="Document Search Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'selected_source_tags' not in st.session_state:
    st.session_state.selected_source_tags = []
if 'used_source_tags' not in st.session_state:
    st.session_state.used_source_tags = []
if 'used_content_tags' not in st.session_state:
    st.session_state.used_content_tags = []
if 'source_tags_method' not in st.session_state:
    st.session_state.source_tags_method = ""

# ===== SETUP AND CONFIGURATION =====
@st.cache_resource
def init_gemini():
    genai.configure(api_key="AIzaSyA1LoiqqFTfZBTOTRujKcaWdUeOKZQ-lX4")
    return genai.GenerativeModel("gemini-1.5-flash")

@st.cache_resource
def init_mongodb():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["mydb"]
    collection = db["gen-tags"]
    return collection

model = init_gemini()
collection = init_mongodb()

# ===== TAG MANAGEMENT FUNCTIONS =====
@st.cache_data(ttl=3600)
def get_predefined_source_tags():
    """Get predefined source tags for filtering"""
    predefined_tags = ["github", "calendar", "confluence", "notion", "zoho", "asana"]
    return predefined_tags

def get_all_source_and_content_tags():
    """Extract source and content tags separately from MongoDB collection"""
    source_tags = []
    content_tags = []
    
    # Get all documents and extract from source_tags and content_tags fields
    for doc in collection.find():
        # Extract from source_tags (list of strings)
        if "source_tags" in doc and isinstance(doc["source_tags"], list):
            source_tags.extend(doc["source_tags"])
        
        # Extract from content_tags (list of objects with "tag" field)
        if "content_tags" in doc and isinstance(doc["content_tags"], list):
            for item in doc["content_tags"]:
                if isinstance(item, dict) and "tag" in item:
                    content_tags.append(item["tag"])
    
    # Clean and deduplicate tags
    def clean_tags(tags):
        cleaned = []
        for tag in tags:
            tag = str(tag).strip().lower()
            if (len(tag) > 2 and 
                tag not in ['and', 'the', 'for', 'with', 'from', 'into', 'that', 'this', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should']):
                cleaned.append(tag)
        return list(set(cleaned))
    
    return clean_tags(source_tags), clean_tags(content_tags)

def get_source_tags_only():
    """Extract only source tags from MongoDB collection"""
    source_tags = []
    
    # Get all documents and extract from source_tags field only
    for doc in collection.find({}, {"source_tags": 1}):
        if "source_tags" in doc and isinstance(doc["source_tags"], list):
            source_tags.extend(doc["source_tags"])
    
    # Clean and deduplicate tags
    def clean_tags(tags):
        cleaned = []
        for tag in tags:
            tag = str(tag).strip().lower()
            if (len(tag) > 2 and 
                tag not in ['and', 'the', 'for', 'with', 'from', 'into', 'that', 'this', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should']):
                cleaned.append(tag)
        return list(set(cleaned))
    
    return clean_tags(source_tags)

def get_content_tags_only():
    """Extract only content tags from MongoDB collection"""
    content_tags = []
    
    # Get all documents and extract from content_tags field only
    for doc in collection.find({}, {"content_tags": 1}):
        if "content_tags" in doc and isinstance(doc["content_tags"], list):
            for item in doc["content_tags"]:
                if isinstance(item, dict) and "tag" in item:
                    content_tags.append(item["tag"])
    
    # Clean and deduplicate tags
    def clean_tags(tags):
        cleaned = []
        for tag in tags:
            tag = str(tag).strip().lower()
            if (len(tag) > 2 and 
                tag not in ['and', 'the', 'for', 'with', 'from', 'into', 'that', 'this', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should']):
                cleaned.append(tag)
        return list(set(cleaned))
    
    return clean_tags(content_tags)

# ===== AI TAG GENERATION FUNCTIONS =====
def get_content_tags_from_gemini(query, content_tags):
    """Generate only content tags from Gemini based on user query"""
    print(f"🔖 DEBUG - Generating content tags for query: '{query}'")
    print(f"🔖 DEBUG - Available content tags: {content_tags}")
    
    prompt = f"""
You are a content tag matching assistant. Given a user query and a list of available content tags, your job is to:
Select the most relevant CONTENT tags from the content tags list that match the user's query.
If the User query exists in available content tags, strictly return it as a tag.
User Query: "{query}"

Available Content Tags: {content_tags}

Return your response in this exact JSON format:
{{
    "content_tags": ["tag1", "tag2", "tag3"]
}}

Rules:
- Only use tags that exist in the provided content tags list
- Content tags should be 3-5 most relevant tags for the query content from the content tags list
- Do not create new tags, only select from existing ones
"""    
    response = model.generate_content(prompt)
    
    try:
        # Clean the response text
        response_text = response.text.strip()
        print(f"🔖 DEBUG - Raw Gemini response for content tags: '{response_text}'")
        
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
        
        result = json.loads(response_text)
        generated_tags = result.get("content_tags", [])
        print(f"🔖 DEBUG - Parsed content tags: {generated_tags}")
        return generated_tags
    except Exception as e:
        print(f"❌ DEBUG - Error processing Gemini response for content tags: {e}")
        st.error(f"Error processing Gemini response for content tags: {e}")
        return []

# ===== DOCUMENT FILTERING FUNCTIONS =====
def filter_documents_with_mongodb_query(source_tags):
    """Filter documents using MongoDB query with source tags"""
    if not source_tags:
        return []
    try:
        conditions = []
        
        for tag in source_tags:
            # Create multiple matching conditions for each tag
            tag_conditions = [
                # 1. Exact match in source_tags array (highest priority)
                {"source_tags": tag},
                
                # 2. Substring match in source_tags array elements
                # This is where the "in" operator concept is applied
                {"source_tags": {"$regex": tag, "$options": "i"}},
                
                # 3. Substring match in document title 
                {"title": {"$regex": tag, "$options": "i"}},
                
                # 4. Substring match in document content/description
                {"content": {"$regex": tag, "$options": "i"}},
                
                
                # 5. If you have other fields, add them here
                # {"description": {"$regex": tag, "$options": "i"}},
                # {"keywords": {"$regex": tag, "$options": "i"}},
            ]
            
            conditions.extend(tag_conditions)
        
        # MongoDB query using $or to match any condition
        # This implements the "in" operator concept - if tag is found anywhere, include the document
        query = {"$or": conditions}
        
        # Execute the enhanced query
        # Replace 'collection' with your actual MongoDB collection variable
        filtered_docs = collection.find(query)
        return list(filtered_docs)
        
    except Exception as e:
        print(f"Error in enhanced MongoDB filtering: {e}")
        return []

def filter_by_content_tags(documents, content_tags):
    """Further filter documents based on content tags"""
    if not content_tags or not documents:
        return documents
    
    try:
        filtered_docs = []
        
        for doc in documents:
            should_include = False
            
            # Check each content tag against the document
            for tag in content_tags:
                # Convert tag to lowercase for case-insensitive matching
                tag_lower = tag.lower()
                
                # Check multiple fields in the document for substring matches
                fields_to_check = [
                    'content_tags',  # Main content tags field
                    'title',         # Document title
                    'content',       # Document content/body
                    'description',   # Document description
                    'keywords',      # Keywords field
                    'category',      # Category field
                    'tags'           # General tags field
                ]
                
                for field in fields_to_check:
                    if field in doc and doc[field]:
                        field_value = doc[field]
                        
                        # Handle different field types
                        if isinstance(field_value, list):
                            # For array fields (like content_tags, keywords)
                            for item in field_value:
                                if isinstance(item, str) and tag_lower in item.lower():
                                    should_include = True
                                    break
                        elif isinstance(field_value, str):
                            # For string fields (like title, content, description)
                            if tag_lower in field_value.lower():
                                should_include = True
                                break
                    
                    if should_include:
                        break
                
                if should_include:
                    break
            
            if should_include:
                filtered_docs.append(doc)
        
        return filtered_docs
        
    except Exception as e:
        print(f"Error in enhanced content tag filtering: {e}")
        return documents

# ===== DOCUMENT SCORING FUNCTIONS =====
def calculate_document_relevance_score(user_query, document):
    """Use Gemini API to calculate a relevance score for a document based on user query and tag scores"""
    # Extract content tags with scores
    content_tags_with_scores = []
    if "content_tags" in document and isinstance(document["content_tags"], list):
        for item in document["content_tags"]:
            if isinstance(item, dict) and "tag" in item and "score" in item:
                content_tags_with_scores.append({
                    "tag": item["tag"],
                    "score": item["score"]
                })
    
    # Extract source tags (these don't have scores typically)
    source_tags = document.get("source_tags", [])    
    if not content_tags_with_scores:
        return 0.0
    
    prompt = f"""
You are a document relevance analyzer. Your task is to calculate a cumulative relevance score for a document based on:
1. The user's query
2. The document's content tags and their individual scores
3. The document's source tags

USER QUERY: "{user_query}"

DOCUMENT CONTENT TAGS WITH SCORES:
{content_tags_with_scores}

DOCUMENT SOURCE TAGS:
{source_tags}

INSTRUCTIONS:
- Analyze how well the document's tags match the user's query intent
- Consider both the relevance of individual tags and their confidence scores
- Higher tag scores indicate higher confidence in that tag's presence in the document
- Source tags provide context about the document's origin/type
- Calculate a cumulative relevance score by combining the individual tag scores based on their relevance to the query
- The score should reflect the sum of relevant tag contributions, not limited to any maximum value
- More relevant tags with higher scores should contribute more to the cumulative score

Return ONLY a single number representing the cumulative relevance score (e.g., 2.45, 5.67, 0.23)
"""
    
    try:
        response = model.generate_content(prompt)
        score_text = response.text.strip()
        # Extract numeric score from response
        score_match = re.search(r'(\d+\.?\d*)', score_text)
        if score_match:
            score = float(score_match.group(1))
            # Return the cumulative score without limiting to 0.0-1.0 range
            return max(0.0, score)
        else:
            return 0.0
            
    except Exception as e:
        st.error(f"Error calculating relevance score: {e}")
        return 0.0

def score_filtered_documents(user_query, filtered_documents):
    """Calculate relevance scores for all filtered documents based on user query"""
    if not filtered_documents:
        return []
    
    scored_documents = []
    
    for doc in filtered_documents:
        # Calculate relevance score
        relevance_score = calculate_document_relevance_score(user_query, doc)
        
        # Add score to document copy
        scored_doc = doc.copy()
        scored_doc['relevance_score'] = relevance_score
        scored_documents.append(scored_doc)
    
    # Sort by relevance score (highest first)
    scored_documents.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return scored_documents

# ===== MAIN SEARCH FUNCTION =====
def search_documents(user_query, selected_source_tags=None):
    """Main search function that returns top 3 most relevant documents
    
    If source tags are provided by user, filters by both source and content tags.
    If no source tags provided, filters only by content tags.
    """
    try:
        # Debug: Print user query
        print(f"🔍 DEBUG - User Query: '{user_query}'")
          # Determine which source tags to use
        if selected_source_tags and len(selected_source_tags) > 0:
            # Use user-selected source tags
            final_source_tags = selected_source_tags
            source_tags_method = "user_selected"
            print(f"🏷️ DEBUG - Using User-Selected Source Tags: {final_source_tags}")
            
            # Filter documents using MongoDB query with source tags
            filtered_documents = filter_documents_with_mongodb_query(final_source_tags)
            print(f"📄 DEBUG - Documents after source filtering: {len(filtered_documents)}")
        else:
            # No source tags provided, skip source filtering
            final_source_tags = []
            source_tags_method = "content_only"
            print("🏷️ DEBUG - No source tags provided, filtering with content tags only")
              # Get all documents when no source filtering is applied
            filtered_documents = list(collection.find())
            print(f"📄 DEBUG - Total documents (no source filtering): {len(filtered_documents)}")
        
        if not filtered_documents:
            print("⚠️ DEBUG - No documents found")
            return {
                "results": [],
                "source_tags_used": final_source_tags,
                "source_tags_method": source_tags_method,
                "content_tags_used": []
            }
        
        # Generate content tags for further filtering - only get content tags when needed
        content_tags = get_content_tags_only()
        matching_content_tags = get_content_tags_from_gemini(user_query, content_tags)
        print(f"📝 DEBUG - AI Generated Content Tags: {matching_content_tags}")
          # Apply content tag filtering to the already MongoDB-filtered documents
        if matching_content_tags:
            content_filtered_documents = filter_by_content_tags(filtered_documents, matching_content_tags)
            print(f"📄 DEBUG - Documents after content filtering: {len(content_filtered_documents)}")
        else:
            content_filtered_documents = filtered_documents
            print("📄 DEBUG - No content tag filtering applied")
        
        # Debug: Show final filtered documents with their tags
        print(f"🔍 DEBUG - Final filtered documents ({len(content_filtered_documents)}):")
        for i, doc in enumerate(content_filtered_documents[:5]):  # Show first 5 documents
            doc_id = doc.get('_id', 'N/A')
            source_tags = doc.get('source_tags', [])
            content_tags = []
            if "content_tags" in doc and isinstance(doc["content_tags"], list):
                for item in doc["content_tags"]:
                    if isinstance(item, dict) and "tag" in item:
                        content_tags.append(item["tag"])
            print(f"  {i+1}. Doc ID: {doc_id}")
            print(f"     Source Tags: {source_tags}")
            print(f"     Content Tags: {content_tags[:5]}{'...' if len(content_tags) > 5 else ''}")
        
        # Score documents based on relevance to user query
        scored_documents = score_filtered_documents(user_query, content_filtered_documents)
        print(f"🎯 DEBUG - Scored documents: {len(scored_documents)}")
          # Return top 3 documents with metadata
        top_3 = scored_documents[:3]
        print(f"🏆 DEBUG - Returning top {len(top_3)} documents")
        
        return {
            "results": top_3,
            "source_tags_used": final_source_tags,
            "source_tags_method": source_tags_method,
            "content_tags_used": matching_content_tags        }
        
    except Exception as e:
        st.error(f"Error during search: {e}")
        return {
            "results": [],
            "source_tags_used": [],
            "source_tags_method": "error",
            "content_tags_used": []
        }

# ===== STREAMLIT UI =====
def main():
    st.title("🔍 Document Search Assistant")
    st.markdown("Enter your search query to find the most relevant documents")
    
    # Get predefined source tags for the checkboxes
    source_tags = get_predefined_source_tags()
    
    # Source tag selection
    if source_tags:
        st.markdown(f"### 🏷️ Filter by Source Tags ({len(source_tags)} available)")
        st.markdown("Select one or more source tags to filter your search (leave empty for content-only filtering):")
        
        # Select All / Clear All buttons
        col_a, col_b, col_c = st.columns([1, 1, 4])
        with col_a:
            if st.button("Select All Tags"):
                for tag in source_tags:
                    st.session_state[f"source_tag_{tag}"] = True
                st.rerun()
        with col_b:
            if st.button("Clear All Tags"):
                for tag in source_tags:
                    st.session_state[f"source_tag_{tag}"] = False
                st.session_state.selected_source_tags = []
                st.rerun()
        
        # Create columns for better layout
        num_cols = 3
        cols = st.columns(num_cols)
        
        # Clear previous selections if tags changed
        if 'selected_source_tags' not in st.session_state:
            st.session_state.selected_source_tags = []
        
        selected_tags = []
        for i, tag in enumerate(source_tags):
            col_idx = i % num_cols
            with cols[col_idx]:
                if st.checkbox(tag, key=f"source_tag_{tag}"):
                    selected_tags.append(tag)
        
        # Update session state
        st.session_state.selected_source_tags = selected_tags
          # Show selected tags
        if selected_tags:
            st.success(f"Selected Source Tags ({len(selected_tags)}): {', '.join(selected_tags)}")
        else:
            st.info("No source tags selected - will filter using content tags only")
        
        st.markdown("---")
    
    # Search bar
    search_query = st.text_input(
        "Search Documents:",
        placeholder="Enter your question or keywords...",
        key="search_input"
    )
      # Search button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_clicked = st.button("🔍 Search", type="primary")
    with col2:        
        if st.button("🗑️ Clear"):
            st.session_state.search_results = []
            st.session_state.last_query = ""
            st.session_state.selected_source_tags = []
            st.session_state.used_source_tags = []
            st.session_state.used_content_tags = []
            st.session_state.source_tags_method = ""
            st.rerun()
      # Perform search when button is clicked
    if search_clicked and search_query:
        st.session_state.last_query = search_query
        
        with st.spinner("Searching for relevant documents..."):
            # Pass selected source tags to search function
            selected_source_tags = st.session_state.get('selected_source_tags', [])
            search_result = search_documents(search_query, selected_source_tags)
              # Store results and metadata
            st.session_state.search_results = search_result.get("results", [])
            st.session_state.used_source_tags = search_result.get("source_tags_used", [])
            st.session_state.source_tags_method = search_result.get("source_tags_method", "")
            st.session_state.used_content_tags = search_result.get("content_tags_used", [])
      # Display results
    if st.session_state.search_results:
        st.markdown("---")
        st.markdown(f"### Results for: *\"{st.session_state.last_query}\"*")          # Show which source tags were used
        used_source_tags = st.session_state.get('used_source_tags', [])
        used_content_tags = st.session_state.get('used_content_tags', [])
        source_method = st.session_state.get('source_tags_method', '')
        
        if source_method == "user_selected":
            st.markdown(f"**🏷️ Filtered by Source Tags:** {', '.join(used_source_tags)}")
        elif source_method == "content_only":
            st.markdown("**🔍 Filtering:** Content tags only (no source tags)")
        else:
            st.markdown("**🏷️ Source Tags:** None used")
        
        # Show the generated content tags
        if used_content_tags:
            st.markdown(f"**🤖 AI-Generated Content Tags:** {', '.join(used_content_tags)}")
        else:
            st.markdown("**🔖 Content Tags:** None generated")
        
        st.markdown(f"Found **{len(st.session_state.search_results)}** most relevant documents:")
        
        for i, doc in enumerate(st.session_state.search_results, 1):
            display_document(doc, i)
    
    elif st.session_state.last_query:
        st.markdown("---")
        st.info("No relevant documents found. Try rephrasing your query or using different keywords.")

def display_document(doc, rank):
    """Display a single document in a nice format"""
    relevance_score = doc.get('relevance_score', 0.0)
    
    # Color coding for score ranges
    if relevance_score >= 0.8:
        score_color = "🟢"
        score_label = "High"
    elif relevance_score >= 0.6:
        score_color = "🟡"
        score_label = "Medium"
    elif relevance_score >= 0.4:
        score_color = "🟠"
        score_label = "Low-Medium"
    else:
        score_color = "🔴"
        score_label = "Low"
    
    with st.expander(f"{score_color} **Rank #{rank}** - Relevance: {score_label} ({relevance_score:.3f})", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Document ID:** `{doc.get('_id', 'N/A')}`")
            
            # Show source tags
            if "source_tags" in doc and doc["source_tags"]:
                st.markdown(f"**🏷️ Source Tags:** {', '.join(doc['source_tags'])}")
            
            # Show content tags with scores
            if "content_tags" in doc and doc["content_tags"]:
                content_tags_with_scores = []
                for item in doc["content_tags"]:
                    if isinstance(item, dict) and "tag" in item:
                        if "score" in item:
                            content_tags_with_scores.append(f"{item['tag']} (score: {item['score']})")
                        else:
                            content_tags_with_scores.append(item["tag"])
                if content_tags_with_scores:
                    st.markdown(f"**📝 Content Tags:** {', '.join(content_tags_with_scores)}")
            
            # Show other fields
            other_fields = {k: v for k, v in doc.items() 
                           if k not in ['_id', 'source_tags', 'content_tags', 'relevance_score']}
            
            if other_fields:
                st.markdown("**📋 Additional Information:**")
                for field, value in other_fields.items():
                    if isinstance(value, str):
                        if len(value) > 200:
                            st.text_area(f"{field}:", value, height=100, disabled=True)
                        else:
                            st.markdown(f"**{field}:** {value}")
                    else:
                        st.markdown(f"**{field}:** {value}")
        
        with col2:
            st.metric("Relevance Score", f"{relevance_score:.3f}")

    st.markdown("---")

if __name__ == "__main__":
    main()
