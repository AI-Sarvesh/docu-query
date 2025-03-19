import os
import json
import asyncio
import concurrent.futures
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import WebSocket
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import re

from config import (
    UPLOAD_DIR, INDEXES_DIR, OPENAI_API_KEY, logger,
    MAX_SESSIONS
)
from utils import (
    extract_entities, extract_key_phrases, get_word_frequencies,
    expand_query, generate_structured_query
)
from models import DebugMode

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}

# Store conversation sessions with TTL (time to live)
conversation_chains = {}
last_accessed = {}  # Track when each session was last accessed

# Store document summaries
document_summaries = {}

# Store processing tasks
processing_tasks = {}

# Store document comparisons
document_comparisons = {}

# Store document visualizations
document_visualizations = {}

# Store document texts (to avoid re-reading files)
document_texts = {}

# Store document chunks (to avoid re-chunking)
document_chunks = {}

# Store debug information
debug_info = {}

# Define custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions about documents with high accuracy and detail.
Use the following pieces of retrieved context to answer the question.
If the exact answer appears in the context, make sure to include it in your response.

Pay special attention to document metadata like:
- Author/writer name
- Document title
- Publication date
- Document topics and subjects

If the question is about who wrote the document, when it was written, or what it's titled, make sure to directly answer with the metadata if it's available in the context.

If you don't know the answer based on the provided context, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

Context:
{context}

Question: {question}

Your answer should be detailed and informative. If the answer is explicitly stated in the context, quote it directly.
"""

# Define summarization prompt
SUMMARY_PROMPT_TEMPLATE = """
Please provide a concise summary of the following document. Focus on the main topics, key points, and important conclusions.
Keep the summary to 2-3 paragraphs.

Document content:
{document_text}
"""

class WebSocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        
    async def on_llm_new_token(self, token: str, **kwargs):
        await self.websocket.send_text(json.dumps({
            "type": "token",
            "content": token
        }))

def cleanup_old_sessions():
    """Remove oldest sessions if we exceed the maximum number of sessions"""
    if len(conversation_chains) <= MAX_SESSIONS:
        return
    
    sorted_sessions = sorted(last_accessed.items(), key=lambda x: x[1])
    sessions_to_remove = len(conversation_chains) - MAX_SESSIONS
    
    for i in range(sessions_to_remove):
        session_id = sorted_sessions[i][0]
        if session_id in conversation_chains:
            del conversation_chains[session_id]
        if session_id in last_accessed:
            del last_accessed[session_id]
        if session_id in document_summaries:
            del document_summaries[session_id]
        if session_id in document_visualizations:
            del document_visualizations[session_id]
        
        logger.info(f"Cleaned up session {session_id} due to memory constraints")
    
    cleanup_document_caches()

def update_session_access(session_id: str):
    """Update the last accessed time for a session"""
    last_accessed[session_id] = datetime.now()

def build_chain(session_id: str, websocket: Optional[WebSocket] = None):
    """Build and return conversation chains for a session"""
    index_path = os.path.join(INDEXES_DIR, session_id)
    if not os.path.exists(index_path):
        return None
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 64}
    )
    
    docsearch = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    try:
        # Replace asyncio.run with direct synchronous access to document chunks
        text_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_extracted.txt")
        if not os.path.exists(text_file_path):
            raise FileNotFoundError(f"Document text file not found for session {session_id}")
        
        with open(text_file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        from fast_chunker import FastChunker
        chunker = FastChunker(
            chunk_size=1000,
            chunk_overlap=200,
            use_spacy=False,
            use_semantic_paragraphs=True
        )
        
        chunks = chunker.chunk_document(text)
        
        bm25_retriever = BM25Retriever.from_texts(chunks)
        bm25_retriever.k = 5
        
        vector_retriever = docsearch.as_retriever(search_kwargs={"k": 5})
        
        retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
    except Exception as e:
        logger.error(f"Error creating hybrid retriever: {str(e)}")
        retriever = docsearch.as_retriever(search_kwargs={"k": 8})
    
    regular_llm = ChatOpenAI(
        temperature=0, 
        model_name="gpt-4o-mini", 
        api_key=OPENAI_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
    
    def format_docs(docs):
        return "\n\n".join([f"Document section {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    def get_retriever_wrapper(question):
        chat_history = memory.load_memory_variables({}).get("chat_history", "")
        doc_summary = document_summaries.get(session_id, "")
        
        if session_id in debug_info and debug_info[session_id].enabled:
            debug_info[session_id].last_query = question
            debug_info[session_id].retrieved_docs = []
            debug_info[session_id].query_variations = []
        
        try:
            # Use the enhanced metadata-aware retriever
            docs = get_history_aware_retriever(question, chat_history, retriever, session_id, doc_summary)
            
            unique_docs = []
            seen_contents = set()
            
            for doc in docs:
                if doc.page_content not in seen_contents:
                    unique_docs.append(doc)
                    seen_contents.add(doc.page_content)
            
            try:
                # Only rerank if these aren't metadata docs
                if not any(doc.metadata.get("source") == "document_metadata" for doc in unique_docs):
                    reranked_docs = rerank_docs(question, unique_docs, top_k=5)
                else:
                    reranked_docs = unique_docs
                
                if session_id in debug_info and debug_info[session_id].enabled:
                    debug_info[session_id].retrieved_docs = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": i
                        }
                        for i, doc in enumerate(reversed(reranked_docs))
                    ]
                
                return reranked_docs
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
                
                if session_id in debug_info and debug_info[session_id].enabled:
                    debug_info[session_id].retrieved_docs = [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "score": 0
                        }
                        for doc in unique_docs[:5]
                    ]
                
                return unique_docs[:5]
        except Exception as e:
            logger.error(f"Error in advanced retrieval: {str(e)}")
            # Fallback to basic retrieval
            docs = retriever.invoke(question)
            
            if session_id in debug_info and debug_info[session_id].enabled:
                debug_info[session_id].retrieved_docs = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": 0
                    }
                    for doc in docs
                ]
            
            return docs
    
    regular_chain = (
        {"context": lambda question: format_docs(get_retriever_wrapper(question)), "question": RunnablePassthrough()}
        | prompt
        | regular_llm
        | StrOutputParser()
    )
    
    streaming_chain = None
    if websocket:
        callback_handler = WebSocketCallbackHandler(websocket)
        
        streaming_llm = ChatOpenAI(
            temperature=0, 
            model_name="gpt-4o-mini", 
            api_key=OPENAI_API_KEY,
            streaming=True,
            callbacks=[callback_handler]
        )
        
        streaming_chain = (
            {"context": lambda question: format_docs(get_retriever_wrapper(question)), "question": RunnablePassthrough()}
            | prompt
            | streaming_llm
            | StrOutputParser()
        )
    
    conversation_chains[session_id] = {
        "chain": regular_chain,
        "streaming_chain": streaming_chain,
        "memory": memory,
        "retriever": retriever
    }
    
    update_session_access(session_id)
    cleanup_old_sessions()
    
    return conversation_chains[session_id]

def extract_text_from_pdf(doc, page_num):
    page = doc.load_page(page_num)
    return page.get_text("text")

async def process_pdf(file_path: str, session_id: str, websocket: WebSocket = None, use_advanced_nlp: bool = False):
    try:
        extracted_text = ""
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "status",
                "content": f"Starting to process PDF with {total_pages} pages..."
            }))
        
        page_texts = []
        page_metadata = []
        
        # Extract document metadata from the first few pages
        document_metadata = {
            "title": "",
            "author": "",
            "date": "",
            "keywords": [],
            "processing_options": {
                "use_advanced_nlp": use_advanced_nlp
            }
        }
        
        # Try to get metadata from PDF properties first
        pdf_metadata = doc.metadata
        if pdf_metadata:
            if pdf_metadata.get("title"):
                document_metadata["title"] = pdf_metadata.get("title")
            if pdf_metadata.get("author"):
                document_metadata["author"] = pdf_metadata.get("author")
            if pdf_metadata.get("creationDate"):
                document_metadata["date"] = pdf_metadata.get("creationDate")
            if pdf_metadata.get("keywords"):
                document_metadata["keywords"] = pdf_metadata.get("keywords").split(",")
        
        # Process the first few pages to extract potential metadata
        first_page_text = ""
        for page_num in range(min(3, total_pages)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_num == 0:
                first_page_text = page_text
        
        # Try to extract title and author from first page text if not found in PDF metadata
        if not document_metadata["title"] or not document_metadata["author"]:
            # Common patterns for title and author
            title_patterns = [
                r"(?i)title[:\s]+([^\n]+)",
                r"(?i)^\s*([^\n]{10,150})\s*$",  # First line that's reasonably long
                r"(?i)report on[:\s]+([^\n]+)",
                r"(?i)([^\n]{10,150})\s*\nby\s+",  # Text followed by "by"
            ]
            
            author_patterns = [
                r"(?i)author[:\s]+([^\n]+)",
                r"(?i)by[:\s]+([^\n]+)",
                r"(?i)prepared by[:\s]+([^\n]+)",
                r"(?i)submitted by[:\s]+([^\n]+)",
                r"(?i)written by[:\s]+([^\n]+)"
            ]
            
            # Try to extract title
            if not document_metadata["title"]:
                for pattern in title_patterns:
                    match = re.search(pattern, first_page_text)
                    if match:
                        document_metadata["title"] = match.group(1).strip()
                        break
            
            # Try to extract author
            if not document_metadata["author"]:
                for pattern in author_patterns:
                    match = re.search(pattern, first_page_text)
                    if match:
                        document_metadata["author"] = match.group(1).strip()
                        break
        
        # Store document metadata
        metadata_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_doc_metadata.json")
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(document_metadata, f)
        
        # Continue with regular processing
        batch_size = 10
        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            batch_pages = list(range(batch_start, batch_end))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 4)) as executor:
                futures = [executor.submit(extract_text_from_pdf, doc, page_num) for page_num in batch_pages]
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    page_num = batch_pages[i]
                    page_text = future.result()
                    page_texts.append((page_num, page_text))
                    
                    page = doc.load_page(page_num)
                    blocks = page.get_text("dict")["blocks"]
                    headings = []
                    
                    for block in blocks:
                        if "lines" in block:
                            for line in block["lines"]:
                                if "spans" in line:
                                    for span in line["spans"]:
                                        if span.get("size", 0) > 12 or span.get("flags", 0) & 2 > 0:
                                            text = span.get("text", "").strip()
                                            if text and len(text) < 100:
                                                headings.append(text)
                    
                    page_metadata.append({
                        "page_num": page_num,
                        "headings": headings
                    })
                    
                    if websocket:
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "content": f"Processing page {page_num + 1} of {total_pages}..."
                        }))
                        await asyncio.sleep(0.05)
        
        page_texts.sort(key=lambda x: x[0])
        extracted_text = "\n\n".join([text for _, text in page_texts])
        
        text_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_extracted.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write(extracted_text)
        
        metadata_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_metadata.json")
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(page_metadata, f)
        
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "status",
                "content": "Text extraction complete. Starting document chunking..."
            }))
        
        doc_structure = "unknown"
        headings_count = sum(len(page["headings"]) for page in page_metadata)
        
        if headings_count > total_pages * 0.5:
            doc_structure = "structured"
        else:
            doc_structure = "narrative"
            
        logger.info(f"Detected document structure: {doc_structure}")
        
        chunk_size = 800
        chunk_overlap = 200
        use_semantic_paragraphs = True
        
        if doc_structure == "structured":
            chunk_size = 600
            chunk_overlap = 150
        
        from fast_chunker import FastChunker
        chunker = FastChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_spacy=use_advanced_nlp,
            use_semantic_paragraphs=use_semantic_paragraphs
        )
        chunks = chunker.chunk_document(extracted_text)
        
        chunks_with_metadata = []
        current_page = 0
        current_headings = []
        
        for i, chunk in enumerate(chunks):
            chunk_start = chunk[:50].strip()
            best_page = 0
            
            for page_num, page_text in page_texts:
                if chunk_start in page_text:
                    best_page = page_num
                    break
            
            page_headings = []
            for meta in page_metadata:
                if meta["page_num"] == best_page:
                    page_headings = meta["headings"]
                    break
            
            chunk_meta = {
                "chunk_id": i,
                "source": f"{session_id}_chunk_{i}",
                "page": best_page + 1,
                "headings": page_headings,
                "doc_structure": doc_structure
            }
            
            chunks_with_metadata.append({
                "content": chunk,
                "metadata": chunk_meta
            })
        
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "status",
                "content": f"Chunking complete. Created {len(chunks)} chunks. Starting vector indexing..."
            }))
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"batch_size": 64}
        )
        
        texts = [item["content"] for item in chunks_with_metadata]
        metadatas = [item["metadata"] for item in chunks_with_metadata]
        docsearch = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
        index_path = os.path.join(INDEXES_DIR, session_id)
        docsearch.save_local(index_path)
        
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "status",
                "content": "Vector indexing complete. Generating document summary..."
            }))
        
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
        
        summary_prompt = f"""
        Please provide a concise summary of the following document. 
        This appears to be a {doc_structure} document with {total_pages} pages.
        
        Focus on the main topics, key points, and important conclusions.
        If you notice any key sections or chapters, please mention them.
        Keep the summary to 2-3 paragraphs.
        
        Document content:
        {{document_text}}
        """
        
        summary_template = ChatPromptTemplate.from_template(summary_prompt)
        
        summarization_text = extracted_text
        if len(extracted_text) > 10000:
            summarization_chunks = [chunks[0]]
            
            if len(chunks) > 4:
                middle_idx = len(chunks) // 2
                summarization_chunks.append(chunks[middle_idx])
            
            if len(chunks) > 2:
                summarization_chunks.append(chunks[-1])
                
            summarization_text = "\n\n".join(summarization_chunks)
        
        summary_chain = summary_template | llm | StrOutputParser()
        summary = await summary_chain.ainvoke({"document_text": summarization_text})
        
        document_summaries[session_id] = summary
        
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "summary",
                "content": summary
            }))
        
        build_chain(session_id, websocket)
        
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "status",
                "content": "Document processing complete. You can now ask questions!"
            }))
        
        if session_id in processing_tasks:
            del processing_tasks[session_id]
            
        return True, "Document processed successfully."
    
    except Exception as e:
        error_message = f"Error processing document: {str(e)}"
        logger.error(error_message)
        if websocket:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": error_message
            }))
        
        if session_id in processing_tasks:
            del processing_tasks[session_id]
            
        return False, error_message

def rerank_docs(query, docs, top_k=5):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in scored_docs[:top_k]]

async def get_document_text(session_id: str) -> str:
    """Get document text, either from cache or by reading the file"""
    if session_id in document_texts:
        return document_texts[session_id]
    
    text_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_extracted.txt")
    if not os.path.exists(text_file_path):
        raise FileNotFoundError(f"Document text file not found for session {session_id}")
    
    with open(text_file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    document_texts[session_id] = text
    return text

async def get_document_chunks(session_id: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Get document chunks, either from cache or by chunking the text"""
    cache_key = f"{session_id}_{chunk_size}_{chunk_overlap}"
    if cache_key in document_chunks:
        return document_chunks[cache_key]
    
    try:
        text = await get_document_text(session_id)
    except FileNotFoundError:
        raise FileNotFoundError(f"Document not found for session {session_id}")
    
    from fast_chunker import FastChunker
    chunker = FastChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        use_spacy=False,
        use_semantic_paragraphs=True
    )
    
    chunks = chunker.chunk_document(text)
    document_chunks[cache_key] = chunks
    return chunks

def cleanup_document_caches():
    """Remove document texts and chunks for sessions that are no longer active"""
    active_sessions = set(conversation_chains.keys())
    
    sessions_to_remove = []
    for session_id in document_texts:
        if session_id not in active_sessions:
            sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del document_texts[session_id]
    
    chunk_keys_to_remove = []
    for cache_key in document_chunks:
        session_id = cache_key.split('_')[0]
        if session_id not in active_sessions:
            chunk_keys_to_remove.append(cache_key)
    
    for cache_key in chunk_keys_to_remove:
        del document_chunks[cache_key]

async def get_document_metadata(session_id: str) -> Dict[str, Any]:
    """Get document metadata, either from cache or by reading the file"""
    metadata_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_doc_metadata.json")
    if os.path.exists(metadata_file_path):
        with open(metadata_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"title": "", "author": "", "date": "", "keywords": []}

def get_history_aware_retriever(query, chat_history, retriever, session_id, doc_summary=""):
    """Enhanced retriever that uses document metadata for specific question types"""
    augmented_query = f"Given the conversation history: {chat_history}\n\nAnswer this question: {query}"
    
    # Check if the query is related to document metadata
    metadata_query_patterns = {
        "author": [r"(?i)who (wrote|authored|created|made|prepared|is the author of) (this|the) (document|paper|report|pdf)",
                  r"(?i)who is the (author|creator|writer)"],
        "title": [r"(?i)what is the (title|name) of (this|the) (document|paper|report|pdf)",
                 r"(?i)(what|how) is (this|the) (document|paper|report|pdf) (called|titled|named)"],
        "date": [r"(?i)when was (this|the) (document|paper|report|pdf) (written|created|prepared|made|published)",
               r"(?i)what is the (date|time) of (this|the) (document|paper|report|pdf)"]
    }
    
    # First, try to get document metadata
    async def get_metadata():
        try:
            return await get_document_metadata(session_id)
        except Exception as e:
            logger.error(f"Error retrieving document metadata: {str(e)}")
            return {}
    
    # Determine if this is a metadata question
    for metadata_type, patterns in metadata_query_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query):
                # This is a metadata question, get the metadata and prepare a custom response
                metadata = asyncio.run(get_metadata())
                
                if metadata.get(metadata_type):
                    # Return a specific document with the metadata as the answer
                    from langchain_core.documents import Document
                    
                    metadata_doc = Document(
                        page_content=f"The {metadata_type} of this document is: {metadata.get(metadata_type)}",
                        metadata={"source": "document_metadata", "metadata_type": metadata_type}
                    )
                    
                    # Also get regular docs for context
                    regular_docs = retriever.invoke(augmented_query)
                    
                    # Put the metadata doc first
                    combined_docs = [metadata_doc] + regular_docs
                    return combined_docs[:5]  # Limit to 5 docs total
    
    # For regular queries, just use the standard retriever
    docs = retriever.invoke(augmented_query)
    return docs 