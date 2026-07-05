# app/models.py
import hashlib
from enum import Enum
from pydantic import BaseModel
from typing import Optional, List


class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict


class DocumentModel(BaseModel):
    page_content: str
    metadata: Optional[dict] = {}

    def generate_digest(self):
        hash_obj = hashlib.md5(self.page_content.encode())
        return hash_obj.hexdigest()


class StoreDocument(BaseModel):
    filepath: str
    filename: str
    file_content_type: str
    file_id: str


class QueryRequestBody(BaseModel):
    query: str
    file_id: str
    k: int = 4
    entity_id: Optional[str] = None
    # Per-request override of HYBRID_SEARCH_ENABLED. None = defer to the env
    # master switch; True/False force hybrid on/off for this call (lets the
    # LibreChat side A/B by user without a second rag-api deployment).
    hybrid: Optional[bool] = None


class CleanupMethod(str, Enum):
    incremental = "incremental"
    full = "full"


class QueryMultipleBody(BaseModel):
    query: str
    file_ids: List[str]
    k: int = 4
    # See QueryRequestBody.hybrid.
    hybrid: Optional[bool] = None


class ContextualizeRequestBody(BaseModel):
    full_text: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
