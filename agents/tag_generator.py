import os
import json
import logging
import signal
import sys
from typing import Dict, List, Optional, TypedDict
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from pymongo.errors import ServerSelectionTimeoutError, WriteError
import pdfplumber
from io import BytesIO
from docx import Document
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from db.mongo_utils import TagMongoDBUtils


# Configure logging to write to a file with timestamp, log level, and message
# Purpose: To track execution details, errors, and progress for debugging and monitoring 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=r'C:\Users\sampa\OneDrive\Documents\Fyndo\tags_gen_log.txt', filemode='w')
logger = logging.getLogger(__name__)

#To maintain and pass state across workflow nodes, ensuring type safety
class TagGenerationState(TypedDict):
    file_keys: List[str]
    current_file_index: int
    file_key: Optional[str]
    file_extension: Optional[str]
    file_content: Optional[str]
    source_tags: List[str]
    content_tags: List[Dict[str, float]]
    description: Optional[str]
    tag_data: Optional[Dict]
    inserted_id: Optional[str]
    error: Optional[str]
    s3_client: boto3.client
    tag_llm: ChatOpenAI
    desc_llm: ChatOpenAI
    mongo_utils: TagMongoDBUtils
    bucket_name: str
    processed_files: set
    force_reprocess: bool

# Connect to S3 and retrieve file keys from the specified bucket
def s3_connect_func(input_path: str) -> tuple[boto3.client, str, List[str]]:
    """Connect to S3 and list objects in the specified bucket root."""
    try:
        load_dotenv()
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if not aws_access_key or not aws_secret_key:
            logger.warning("AWS credentials not found in .env. Attempting IAM role or AWS CLI.")
        s3_client = boto3.client("s3")
        logger.info("Connected to S3")

        s3_parts = input_path.replace("s3://", "").split("/", 1)
        bucket_name = s3_parts[0]
        prefix = ""  # No prefix, list all files in bucket root
        logger.info(f"Listing all objects in bucket: {bucket_name}")
        file_keys = []
        paginator = s3_client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, PaginationConfig={'MaxKeys': 1000})
        for page in page_iterator:
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    if key and not key.endswith("/"):
                        file_keys.append(key)
            else:
                logger.info(f"No objects in bucket {bucket_name}")
        logger.info(f"Total files found in S3 bucket {bucket_name}: {len(file_keys)}")
        return s3_client, bucket_name, file_keys
    except ClientError as e:
        logger.error(f"S3 connection error: {e.response['Error']['Code']} - {e.response['Error']['Message']}")
        raise
    except Exception as e:
        logger.error(f"S3 connection error: {str(e)}")
        raise

# Parse S3 file content and extract source tags
def parse_func(s3_client: boto3.client, bucket_name: str, file_key: str, file_extension: str) -> tuple[str, List[str]]:
    """Parse content and extract source tags from an S3 file."""
    source_tags = []
    content = ""
    try:
        path_components = file_key.split("/")[:-1]
        source_tags.extend([comp.lower() for comp in path_components if comp])
        if file_extension:
            source_tags.append(file_extension.lstrip(".").lower())
        source_tags = list(set(source_tags))

        logger.info(f"Attempting to fetch S3 object: {bucket_name}/{file_key}")
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        raw_content = obj["Body"].read()
        logger.info(f"Fetched {len(raw_content)} bytes for {file_key}")

        if file_extension == ".pdf":
            try:
                pdf_file = BytesIO(raw_content)
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text() or ""
                        content += text
                logger.info(f"Extracted PDF content for {file_key}: {len(content)} chars")
            except Exception as e:
                logger.warning(f"Error extracting PDF content for {file_key}: {e}")
                content = raw_content.decode("utf-8", errors="ignore")
        elif file_extension in [".docx", ".doc"]:
            try:
                doc_file = BytesIO(raw_content)
                doc = Document(doc_file)
                content = "\n".join([para.text for para in doc.paragraphs if para.text])
                logger.info(f"Extracted DOCX/DOC content for {file_key}: {len(content)} chars")
            except Exception as e:
                logger.warning(f"Error extracting DOCX/DOC content for {file_key}: {e}")
                content = raw_content.decode("utf-8", errors="ignore")
        elif file_extension == ".csv":
            try:
                csv_file = BytesIO(raw_content)
                df = pd.read_csv(csv_file, encoding="utf-8", errors="ignore")
                content = df.to_string()
                logger.info(f"Extracted CSV content for {file_key}: {len(content)} chars")
            except Exception as e:
                logger.warning(f"Error extracting CSV content for {file_key}: {e}")
                content = raw_content.decode("utf-8", errors="ignore")
        elif file_extension in [".txt", ".html", ".json"]:
            try:
                content = raw_content.decode("utf-8", errors="ignore")
                logger.info(f"Extracted text content for {file_key}: {len(content)} chars")
            except Exception as e:
                logger.warning(f"Error decoding {file_key} as text: {e}")
                content = ""
        else:
            try:
                content = raw_content.decode("utf-8", errors="ignore")
                logger.info(f"Fallback decoding for {file_extension} in {file_key}: {len(content)} chars")
            except Exception as e:
                logger.warning(f"Unable to decode {file_key} as text: {e}")
                content = ""

        if not content:
            logger.warning(f"No content extracted for {file_key}, using file name as fallback")
            content = file_key

        logger.info(f"Parsed {len(content)} characters from {file_key}, extension: {file_extension}, source_tags: {source_tags}")
        return content, source_tags
    except Exception as e:
        logger.error(f"Error parsing {file_key}: {str(e)}")
        return file_key, ["error"]

# Initialize resources and workflow state
def initialize_resources(input_path: str, session_id: str, force_reprocess: bool) -> TagGenerationState:
    """Initialize S3, LLM, MongoDB, and state for tag generation."""
    try:
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env")

        logger.info("Testing OpenAI API key...")
        test_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_retries=2, api_key=openai_api_key)
        test_llm.invoke("Test query: return 'OK'")
        logger.info("OpenAI API key is valid")

        if not input_path.startswith("s3://bucketdatafyndo.ai/"):
            raise ValueError("Input path must be s3://bucketdatafyndo.ai/")

        s3_client, bucket_name, file_keys = s3_connect_func(input_path)
        
        mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        logger.info(f"Connecting to MongoDB at {mongo_uri}")
        mongo_utils = TagMongoDBUtils(mongo_uri)
        mongo_utils.client.admin.command('ping')
        logger.info(f"Successfully connected to MongoDB at {mongo_uri}")
        db = mongo_utils.client["mydb"]
        collection = db["gen-tags"]
        doc_count = collection.count_documents({})
        logger.info(f"Initial documents in mydb.gen-tags: {doc_count}")

        processed_files = set()
        current_index = 0
        if not force_reprocess:
            existing_tags = mongo_utils.get_all_tags()
            processed_files.update(tag["file_source"] for tag in existing_tags)
            logger.info(f"Loaded {len(existing_tags)} existing tags from MongoDB")
        else:
            logger.info("Force reprocess enabled, ignoring existing tags")

        saved_state = mongo_utils.get_state(session_id)
        if saved_state and not force_reprocess:
            current_index = saved_state.get("current_file_index", 0)
            processed_files.update(saved_state.get("processed_files", []))
            logger.info(f"Resumed state: index={current_index}, processed={len(processed_files)} files")

        tag_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_retries=2, api_key=openai_api_key)
        desc_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_retries=2, api_key=openai_api_key)

        return TagGenerationState(
            file_keys=file_keys,
            current_file_index=current_index,
            file_key=None,
            file_extension=None,
            file_content=None,
            source_tags=[],
            content_tags=[],
            description=None,
            tag_data=None,
            inserted_id=None,
            error=None,
            s3_client=s3_client,
            tag_llm=tag_llm,
            desc_llm=desc_llm,
            mongo_utils=mongo_utils,
            bucket_name=bucket_name,
            processed_files=processed_files,
            force_reprocess=force_reprocess
        )
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise
    
    
# Clean up resources and log final state
def cleanup_resources(state: TagGenerationState):
    """Clean up resources, close MongoDB connection, and log final state."""
    try:
        if state["mongo_utils"]:
            logger.info("Finalizing MongoDB cleanup")
            collection = state["mongo_utils"].collection
            doc_count = collection.count_documents({})
            logger.info(f"Final documents in mydb.gen-tags: {doc_count}")
            state["mongo_utils"].close_connection()
        logger.info(f"Processing complete. Processed {len(state['processed_files'])} files. Tags saved to MongoDB mydb.gen-tags")
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        raise


# Generate content tags using LLM
def generate_tags_node(state: TagGenerationState) -> TagGenerationState:
    try:
        file_key = state["file_keys"][state["current_file_index"]]
        if not file_key:
            logger.error(f"Empty file key at index {state['current_file_index']}")
            return {
                **state,
                "current_file_index": state["current_file_index"] + 1,
                "error": "Empty file key"
            }

        file_source = f"s3://{state['bucket_name']}/{file_key}"
        if not state.get("force_reprocess", False) and state["mongo_utils"].tag_exists(file_source):
            logger.info(f"Tag exists for {file_key}")
            state["processed_files"].add(file_source)
            return {
                **state,
                "current_file_index": state["current_file_index"] + 1,
                "error": None
            }

        file_extension = os.path.splitext(file_key)[1].lower()
        content, source_tags = parse_func(state["s3_client"], state["bucket_name"], file_key, file_extension)

        tag_llm = state["tag_llm"]
        content_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a specialized Tag Generator Agent, responsible for generating content-relevant tags after receiving the fully parsed output of a document. Your task is to:\n\n- Understand the core themes, topics, and intent of the parsed document.\n- Generate a set of accurate, content-grounded tags.\n- Assign a relevancy score to each tag, ranging from 0.00 (barely relevant) to 1.00 (critical to the document’s core meaning).\n\nFocus only on key concepts, tools, and entities present in the content. Do not include synonyms or inferred topics. Output must be in JSON format:\n[{{\"tag\": \"value\", \"score\": 0.75}}, ...]\nAll tags must be lowercase. Do not include any markdown or extra text."),
            ("human", "File: {file_name}\nContent: {content}")
        ])
        content_chain = content_prompt | tag_llm

        def invoke_with_retry(chain, input_data, retries=5):
            for attempt in range(retries):
                try:
                    logger.info(f"Invoking LLM for {input_data['file_name']}, attempt {attempt + 1}")
                    response = chain.invoke({"file_name": input_data["file_name"], "content": input_data["content"][:4000]})
                    logger.info(f"LLM response for {input_data['file_name']}: {str(response.content)[:200]}...")
                    return response
                except Exception as e:
                    logger.warning(f"LLM error for {input_data['file_name']}, attempt {attempt + 1}: {str(e)}")
                    if attempt < retries - 1:
                        continue
                    else:
                        logger.error(f"Failed after {retries} attempts for {input_data['file_name']}: {str(e)}")
                        raise

        content_tags = []
        if content:
            content_response = invoke_with_retry(
                content_chain,
                {"file_name": file_key, "content": content}
            )
            response_text = content_response.content.strip()
            logger.info(f"Tag response for {file_key}: {response_text}")
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            try:
                content_tags = json.loads(response_text)
                content_tags = [{"tag": tag["tag"].lower(), "score": min(max(float(tag["score"]), 0), 1)} for tag in content_tags if "tag" in tag and "score" in tag]
                logger.info(f"Generated content tags for {file_key}: {content_tags}")
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response for {file_key}: {response_text[:200]}... Error: {e}")
                content_tags = []

        return {
            **state,
            "file_key": file_key,
            "file_extension": file_extension,
            "file_content": content,
            "source_tags": source_tags,
            "content_tags": content_tags,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error generating tags for {file_key}: {str(e)}")
        return {
            **state,
            "error": str(e)
        }

# Generate file description and save tags to MongoDB
def generate_description_node(state: TagGenerationState) -> TagGenerationState:
    if state.get("error") or not state.get("file_key"):
        logger.warning(f"Skipping description generation due to error or missing file_key: {state.get('error', 'No file_key')}")
        return {
            **state,
            "current_file_index": state["current_file_index"] + 1,
            "error": state.get("error")
        }

    try:
        file_key = state["file_key"]
        file_source = f"s3://{state['bucket_name']}/{file_key}"
        content = state["file_content"]
        source_tags = state["source_tags"]
        content_tags = state["content_tags"]

        desc_llm = state["desc_llm"]
        desc_prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate a concise description (1-2 sentences) based on file content. Summarize purpose or key details. Return plain text, no markdown."),
            ("human", "File: {file_name}\nContent: {content}")
        ])
        desc_chain = desc_prompt | desc_llm

        def invoke_with_retry(chain, input_data, retries=5):
            for attempt in range(retries):
                try:
                    logger.info(f"Invoking LLM for {input_data['file_name']}, attempt {attempt + 1}")
                    response = chain.invoke({"file_name": input_data["file_name"], "content": input_data["content"][:4000]})
                    logger.info(f"LLM response for {input_data['file_name']}: {str(response.content)[:200]}...")
                    return response
                except Exception as e:
                    logger.warning(f"LLM error for {input_data['file_name']}, attempt {attempt + 1}: {str(e)}")
                    if attempt < retries - 1:
                        continue
                    else:
                        logger.error(f"Failed after {retries} attempts for {input_data['file_name']}: {str(e)}")
                        raise

        description = ""
        if content:
            desc_response = invoke_with_retry(
                desc_chain,
                {"file_name": file_key, "content": content}
            )
            description = desc_response.content.strip()
            logger.info(f"Generated description for {file_key}: {description}")
        else:
            logger.warning(f"No content for description, using fallback for {file_key}")
            description = f"File related to {', '.join(source_tags)}."

        hyperlink = ""
        try:
            s3_client = state["s3_client"]
            presigned_url = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': state["bucket_name"], 'Key': file_key},
                ExpiresIn=3600
            )
            hyperlink = presigned_url
            logger.info(f"Generated presigned URL for {file_key}")
        except Exception as e:
            logger.error(f"Error generating presigned URL for {file_key}: {e}")

        tag_data = {
            "file_name": file_key,
            "file_source": file_source,
            "source_tags": source_tags,
            "content_tags": content_tags,
            "description": description,
            "hyperlink": hyperlink
        }

        try:
            logger.info(f"Attempting to save tag_data for {file_key}: {json.dumps(tag_data, indent=2)}")
            inserted_id = state["mongo_utils"].insert_tag(tag_data)
            if not inserted_id:
                logger.error(f"No ID returned for {file_key} tag insert/update")
            else:
                logger.info(f"Successfully saved tag for {file_key} with ID: {inserted_id}")
                state["processed_files"].add(file_source)
                collection = state["mongo_utils"].collection
                doc_count = collection.count_documents({})
                logger.info(f"Current documents in mydb.gen-tags: {doc_count}")
        except ServerSelectionTimeoutError as e:
            logger.error(f"Database connection timeout for {file_key}: {str(e)}")
            raise
        except WriteError as e:
            logger.error(f"Database write error for {file_key}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error saving tags for {file_key}: {str(e)}")
            raise

        return {
            **state,
            "description": description,
            "hyperlink": hyperlink,
            "tag_data": tag_data,
            "inserted_id": inserted_id,
            "current_file_index": state["current_file_index"] + 1,
            "error": None
        }
    except Exception as e:
        logger.error(f"Error generating description for {file_key}: {str(e)}")
        return {
            **state,
            "current_file_index": state["current_file_index"] + 1,
            "error": str(e)
        }

# Build the tag generation workflow
def build_tag_generator_workflow():
    workflow = StateGraph(TagGenerationState)
    workflow.add_node("generate_tags", generate_tags_node)
    workflow.add_node("generate_description", generate_description_node)
    workflow.add_edge(START, "generate_tags")
    workflow.add_conditional_edges(
        "generate_tags",
        lambda state: "generate_description" if state.get("file_key") else ("generate_tags" if state["current_file_index"] < len(state["file_keys"]) else END),
        {"generate_description": "generate_description", "generate_tags": "generate_tags", END: END}
    )
    workflow.add_conditional_edges(
        "generate_description",
        lambda state: END if state["current_file_index"] >= len(state["file_keys"]) or state.get("error") else "generate_tags",
        {"generate_tags": "generate_tags", END: END}
    )
    return workflow.compile()

# Run the tag generation workflow
def run_tag_generator(input_path: str = "s3://bucketdatafyndo.ai/", session_id: str = "fyndsession", force_reprocess: bool = True):
    logger.info(f"Starting workflow with input_path: {input_path}, force_reprocess: {force_reprocess}")
    try:
        state = initialize_resources(input_path, session_id, force_reprocess)
        
        def signal_handler(sig, frame):
            logger.info("Interrupt received, saving state and closing MongoDB connection...")
            state["mongo_utils"].insert_state({
                "session_id": session_id,
                "current_file_index": state["current_file_index"],
                "processed_files": list(state["processed_files"]),
                "input_path": input_path,
                "force_reprocess": force_reprocess,
                "last_processed_file": state["file_key"]
            })
            logger.info(f"Saved state for session {session_id}")
            cleanup_resources(state)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        workflow = build_tag_generator_workflow()
        final_state = workflow.invoke(state, config={"recursion_limit": 100000})
        cleanup_resources(final_state)
        logger.info("Workflow completed")
        return final_state
    except Exception as e:
        logger.error(f"Workflow error: {str(e)}")
        if "state" in locals():
            cleanup_resources(state)
        return {"error": str(e)}
# Entry point
if __name__ == "__main__":
    run_tag_generator(force_reprocess=False)