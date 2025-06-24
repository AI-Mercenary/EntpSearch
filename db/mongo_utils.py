from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, WriteError, ServerSelectionTimeoutError
from typing import Dict, List, Optional
import logging
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TagMongoDBUtils:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017", max_retries: int = 3, retry_delay: int = 5):
        for attempt in range(max_retries):
            try:
                self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                self.client.admin.command('ping')
                self.db = self.client["mydb"]
                self.collection = self.db["gen-tags"]
                self.state_collection = self.db["gen-tags-state"]
                if "gen-tags" not in self.db.list_collection_names():
                    self.db.create_collection("gen-tags")
                    logger.info("Created mydb.gen-tags collection")
                if "gen-tags-state" not in self.db.list_collection_names():
                    self.db.create_collection("gen-tags-state")
                    logger.info("Created mydb.gen-tags-state collection")
                doc_count = self.collection.count_documents({})
                logger.info(f"Connected to MongoDB at {mongo_uri}, using mydb.gen-tags with {doc_count} documents")
                break
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.error(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to connect to MongoDB after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error during MongoDB connection: {e}")
                raise

    def tag_exists(self, file_source: str) -> bool:
        try:
            exists = bool(self.collection.find_one({"file_source": file_source}))
            logger.info(f"Tag exists for {file_source}: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking tag existence for {file_source}: {e}")
            return False

    def insert_tag(self, tag_data: Dict) -> str:
        try:
            file_source = tag_data["file_source"]
            tag_data["created_at"] = datetime.utcnow()
            logger.info(f"Inserting/updating tag_data for {tag_data['file_name']}: {tag_data}")
            if not self.tag_exists(file_source):
                result = self.collection.insert_one(tag_data)
                doc_count = self.collection.count_documents({})
                logger.info(f"Inserted tag for {tag_data['file_name']} with ID: {result.inserted_id}, total documents: {doc_count}")
                return str(result.inserted_id)
            else:
                self.collection.update_one(
                    {"file_source": file_source},
                    {"$set": {
                        "file_name": tag_data["file_name"],
                        "source_tags": tag_data["source_tags"],
                        "content_tags": tag_data["content_tags"],
                        "description": tag_data["description"],
                        "hyperlink": tag_data["hyperlink"],
                        "created_at": datetime.utcnow()
                    }}
                )
                doc_count = self.collection.count_documents({})
                logger.info(f"Updated tag for {tag_data['file_name']}, total documents: {doc_count}")
                return str(self.collection.find_one({"file_source": file_source})["_id"])
        except WriteError as e:
            logger.error(f"Write error inserting/updating tag for {tag_data.get('file_name', 'unknown')}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error inserting/updating tag for {tag_data.get('file_name', 'unknown')}: {e}")
            raise

    def get_all_tags(self) -> List[Dict]:
        try:
            tags = list(self.collection.find())
            logger.info(f"Retrieved {len(tags)} tags from MongoDB")
            return tags
        except Exception as e:
            logger.error(f"Error retrieving tags: {e}")
            return []

    def insert_state(self, state_data: Dict) -> None:
        try:
            session_id = state_data["session_id"]
            state_data["saved_at"] = datetime.utcnow()
            self.state_collection.update_one(
                {"session_id": session_id},
                {"$set": state_data},
                upsert=True
            )
            logger.info(f"Saved state for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving state for session {session_id}: {e}")
            raise

    def get_state(self, session_id: str) -> Optional[Dict]:
        try:
            state = self.state_collection.find_one({"session_id": session_id})
            logger.info(f"Retrieved state for session {session_id}: {state if state else 'None'}")
            return state
        except Exception as e:
            logger.error(f"Error retrieving state for session {session_id}: {e}")
            return None

    def close_connection(self):
        try:
            self.client.close()
            logger.info("Closed MongoDB connection")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")