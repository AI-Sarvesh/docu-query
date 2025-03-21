import os
import uuid
import shutil
import sqlite3
import json
import re
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import HOST, PORT, UPLOAD_DIR, INDEXES_DIR, OPENAI_API_KEY, logger
from models import (
    QueryRequest, QueryResponse, ProcessingStatus,
    FeedbackRequest, ComparisonRequest, ComparisonResult,
    VisualizationData, DebugMode
)
from services import (
    process_pdf, build_chain, active_connections,
    conversation_chains, processing_tasks,
    document_summaries, document_comparisons,
    document_visualizations, debug_info,
    WebSocketCallbackHandler, rerank_docs, get_document_text,
    CUSTOM_PROMPT_TEMPLATE
)
from utils import (
    count_syllables, extract_entities, extract_key_phrases,
    get_word_frequencies
)

app = FastAPI(title="Document Q&A API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Jinja2 templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize SQLite database for feedback
def init_db():
    conn = sqlite3.connect('feedback.db')
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        query TEXT NOT NULL,
        answer TEXT NOT NULL,
        rating TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.commit()
    conn.close()

init_db()

# **Helper Functions**
async def send_ws_message(websocket: WebSocket, message_type: str, content):
    """Send a JSON message over WebSocket."""
    await websocket.send_text(json.dumps({"type": message_type, "content": content}))

def save_feedback(session_id: str, query: str, answer: str, rating: str) -> tuple[bool, str]:
    """Save feedback to the database and return success status with error message if any."""
    try:
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (session_id, query, answer, rating) VALUES (?, ?, ?, ?)",
            (session_id, query, answer, rating)
        )
        conn.commit()
        conn.close()
        return True, None
    except Exception as e:
        error_msg = f"Error saving feedback: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

async def get_streaming_chain(chain_data, websocket):
    """Get or create a streaming chain for query processing."""
    if "streaming_chain" not in chain_data:
        callback_handler = WebSocketCallbackHandler(websocket)
        streaming_llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            streaming=True,
            callbacks=[callback_handler]
        )
        prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
        retriever = chain_data["retriever"]

        def format_docs(docs):
            return "\n\n".join([f"Document section {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

        def get_history_aware_retriever(query):
            chat_history = chain_data["memory"].load_memory_variables({}).get("chat_history", "")
            augmented_query = f"Given the conversation history: {chat_history}\n\nAnswer this question: {query}"
            docs = retriever.invoke(augmented_query)
            try:
                return rerank_docs(query, docs, top_k=5)
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
                return docs

        streaming_chain = (
            {"context": lambda question: format_docs(get_history_aware_retriever(question)), "question": RunnablePassthrough()}
            | prompt
            | streaming_llm
            | StrOutputParser()
        )
        chain_data["streaming_chain"] = streaming_chain
    return chain_data["streaming_chain"]

async def handle_fallback(query: str, session_id: str) -> str:
    """Handle fallback retrieval when initial response is unhelpful."""
    try:
        doc_text = await get_document_text(session_id)
        fallback_prompt = f"""
        I need to find information about "{query}" in a document.
        
        Here's the specific question: {query}
        
        Please search the document carefully for any relevant information, even if it's not explicitly stated.
        Look for related terms, synonyms, or concepts that might help answer this question.
        If you find anything that might be relevant, please provide that information.
        
        Document summary: {document_summaries.get(session_id, "Not available")}
        """
        fallback_llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
        response = await fallback_llm.invoke(fallback_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.error(f"Error in fallback retrieval: {str(e)}")
        return None

async def fallback_to_regular_chain(chain_data, query: str, websocket: WebSocket) -> str:
    """Fall back to regular chain if streaming fails."""
    try:
        regular_chain = chain_data["chain"]
        result = await regular_chain.ainvoke(query)
        await send_ws_message(websocket, "token", result)
        return result
    except Exception as e:
        error_message = f"Error processing query: {str(e)}"
        logger.error(error_message)
        await send_ws_message(websocket, "token", f"I'm sorry, I encountered an error: {str(e)}")
        return None

# **WebSocket Handlers**
async def handle_start_processing(websocket: WebSocket, session_id: str, message: dict):
    temp_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_temp.pdf")
    use_advanced_nlp = message.get("use_advanced_nlp", False)
    if os.path.exists(temp_file_path):
        processing_task = asyncio.create_task(process_pdf(temp_file_path, session_id, websocket, use_advanced_nlp))
        processing_tasks[session_id] = processing_task
    else:
        await send_ws_message(websocket, "error", "No document found to process.")

async def handle_query(websocket: WebSocket, session_id: str, message: dict):
    query = message["content"]
    if session_id not in conversation_chains:
        chain_data = build_chain(session_id, websocket)
        if not chain_data:
            await send_ws_message(websocket, "error", "Session not found or document not processed yet.")
            return
    else:
        chain_data = conversation_chains[session_id]

    memory = chain_data["memory"]
    memory.save_context({"input": query}, {"output": ""})

    await send_ws_message(websocket, "status", "Thinking...")
    await send_ws_message(websocket, "start_stream", None)

    try:
        streaming_chain = await get_streaming_chain(chain_data, websocket)
        result = await streaming_chain.ainvoke(query)

        dont_know_phrases = ["i don't know", "i do not know", "don't have enough information", "cannot find", "no information"]
        if any(phrase in result.lower() for phrase in dont_know_phrases):
            logger.info(f"Initial response was 'I don't know'. Trying fallback for query: {query}")
            fallback_response = await handle_fallback(query, session_id)
            if fallback_response and not any(phrase in fallback_response.lower() for phrase in dont_know_phrases):
                await send_ws_message(websocket, "token", "\n\nLet me search more deeply in the document...\n\n")
                await send_ws_message(websocket, "token", fallback_response)
                result += "\n\nLet me search more deeply in the document...\n\n" + fallback_response
        memory.chat_memory.messages[-1].content = result
    except Exception as e:
        logger.error(f"Error with streaming response: {str(e)}")
        result = await fallback_to_regular_chain(chain_data, query, websocket)
        if result:
            memory.chat_memory.messages[-1].content = result
    finally:
        await send_ws_message(websocket, "end_stream", None)

async def handle_feedback(websocket: WebSocket, session_id: str, message: dict):
    success, error_msg = save_feedback(session_id, message["query"], message["answer"], message["rating"])
    if success:
        await send_ws_message(websocket, "feedback_received", "Thank you for your feedback!")
    else:
        await send_ws_message(websocket, "error", error_msg)

async def handle_compare_documents(websocket: WebSocket, session_id: str, message: dict):
    session_id1 = message.get("session_id1")
    session_id2 = message.get("session_id2")
    comparison_type = message.get("comparison_type", "semantic")
    if not session_id1 or not session_id2:
        await send_ws_message(websocket, "error", "Both document session IDs are required.")
        return

    await send_ws_message(websocket, "status", "Comparing documents...")
    result = await compare_documents(session_id1, session_id2, comparison_type)
    if "error" in result:
        await send_ws_message(websocket, "error", result["error"])
    else:
        await send_ws_message(websocket, "comparison_result", result)

async def handle_generate_visualization(websocket: WebSocket, session_id: str, message: dict):
    await send_ws_message(websocket, "status", "Generating document visualization...")
    result = await generate_visualization(session_id)
    if "error" in result:
        await send_ws_message(websocket, "error", result["error"])
    else:
        await send_ws_message(websocket, "visualization_result", result)

# **WebSocket Endpoint**
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket

    handlers = {
        "start_processing": handle_start_processing,
        "query": handle_query,
        "feedback": handle_feedback,
        "compare_documents": handle_compare_documents,
        "generate_visualization": handle_generate_visualization,
    }

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            if msg_type in handlers:
                await handlers[msg_type](websocket, session_id, message)
            else:
                await send_ws_message(websocket, "error", "Unknown message type")
    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await send_ws_message(websocket, "error", f"An error occurred: {str(e)}")

# **REST Endpoints**
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/", response_model=ProcessingStatus)
async def upload_document(file: UploadFile = File(...), use_advanced_nlp: bool = False):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        session_id = str(uuid.uuid4())
        temp_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_temp.pdf")
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        if not file.filename.startswith('_ws_'):
            processing_task = asyncio.create_task(process_pdf(temp_file_path, session_id, None, use_advanced_nlp))
            processing_tasks[session_id] = processing_task
        
        return ProcessingStatus(
            status="processing",
            message="Document uploaded. Connect via WebSocket for updates.",
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.get("/status/{session_id}", response_model=ProcessingStatus)
async def check_status(session_id: str, use_advanced_nlp: bool = False):
    if session_id in processing_tasks:
        return ProcessingStatus(status="processing", message="Document is still being processed.", session_id=session_id)
    if session_id in conversation_chains:
        return ProcessingStatus(status="ready", message="Document processing completed.", session_id=session_id)
    
    index_path = os.path.join(INDEXES_DIR, session_id)
    if os.path.exists(index_path):
        try:
            build_chain(session_id)
            return ProcessingStatus(status="ready", message="Document processing completed.", session_id=session_id)
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return ProcessingStatus(status="error", message=f"Error loading document: {str(e)}", session_id=session_id)
    
    temp_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_temp.pdf")
    if os.path.exists(temp_file_path):
        if session_id not in processing_tasks:
            processing_task = asyncio.create_task(process_pdf(temp_file_path, session_id, None, use_advanced_nlp))
            processing_tasks[session_id] = processing_task
        return ProcessingStatus(status="processing", message="Document is being processed.", session_id=session_id)
    
    return ProcessingStatus(status="error", message="Document not found.", session_id=session_id)

@app.get("/summary/{session_id}")
async def get_summary(session_id: str):
    if session_id in document_summaries:
        return {"summary": document_summaries[session_id]}
    raise HTTPException(status_code=404, detail="Summary not found for this session.")

@app.post("/feedback/", status_code=201)
async def submit_feedback(feedback: FeedbackRequest):
    success, error_msg = save_feedback(feedback.session_id, feedback.query, feedback.answer, feedback.rating)
    if success:
        return {"message": "Feedback submitted successfully"}
    raise HTTPException(status_code=500, detail=error_msg)

@app.post("/query/", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    session_id = request.session_id
    if session_id not in conversation_chains:
        chain_data = build_chain(session_id)
        if not chain_data:
            raise HTTPException(status_code=404, detail="Session not found or document not processed yet.")
    else:
        chain_data = conversation_chains[session_id]
    
    chain = chain_data["chain"]
    memory = chain_data["memory"]
    memory.save_context({"input": request.query}, {"output": ""})
    result = chain.invoke(request.query)
    memory.chat_memory.messages[-1].content = result
    return QueryResponse(answer=result, sources=[])

@app.post("/compare/", response_model=ComparisonResult)
async def compare_documents_endpoint(request: ComparisonRequest):
    result = await compare_documents(request.session_id1, request.session_id2, request.comparison_type)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/visualization/{session_id}")
async def get_visualization(session_id: str):
    if session_id in document_visualizations:
        return document_visualizations[session_id]
    result = await generate_visualization(session_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@app.get("/document_stats/{session_id}")
async def get_document_stats(session_id: str):
    try:
        text = await get_document_text(session_id)
        word_count = len(re.findall(r'\b\w+\b', text))
        sentence_count = len(re.findall(r'[.!?]+', text)) or 1
        paragraph_count = len(text.split('\n\n'))
        
        entities = extract_entities(text)
        key_phrases = extract_key_phrases(text, num_phrases=10)
        word_frequencies = dict(list(get_word_frequencies(text, max_words=20).items()))
        
        words = re.findall(r'\b\w+\b', text.lower())
        syllable_count = sum(count_syllables(word) for word in words)
        flesch_score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        flesch_score = max(0, min(100, flesch_score))
        
        readability_level = "Very Difficult"
        if flesch_score >= 90:
            readability_level = "Very Easy"
        elif flesch_score >= 80:
            readability_level = "Easy"
        elif flesch_score >= 70:
            readability_level = "Fairly Easy"
        elif flesch_score >= 60:
            readability_level = "Standard"
        elif flesch_score >= 50:
            readability_level = "Fairly Difficult"
        elif flesch_score >= 30:
            readability_level = "Difficult"
        
        return {
            "basic_stats": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "readability_score": round(flesch_score, 1),
                "readability_level": readability_level
            },
            "top_entities": {
                "people": entities["people"][:5],
                "organizations": entities["organizations"][:5],
                "locations": entities["locations"][:5],
                "dates": entities["dates"][:5]
            },
            "top_phrases": key_phrases[:10],
            "top_words": word_frequencies
        }
    except Exception as e:
        logger.error(f"Error generating document statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating document statistics: {str(e)}")

@app.post("/debug/{session_id}")
async def toggle_debug_mode(session_id: str, enable: bool = True):
    if session_id not in debug_info:
        debug_info[session_id] = DebugMode(session_id=session_id, enabled=enable)
    else:
        debug_info[session_id].enabled = enable
    return {"message": f"Debug mode {'enabled' if enable else 'disabled'} for session {session_id}"}

@app.get("/debug/{session_id}")
async def get_debug_info(session_id: str):
    if session_id not in debug_info:
        raise HTTPException(status_code=404, detail="No debug information available for this session")
    return debug_info[session_id]

# **Utility Functions**
async def compare_documents(session_id1: str, session_id2: str, comparison_type: str = "semantic"):
    try:
        cache_key = f"{session_id1}_{session_id2}_{comparison_type}"
        reverse_cache_key = f"{session_id2}_{session_id1}_{comparison_type}"
        
        if cache_key in document_comparisons:
            return document_comparisons[cache_key]
        if reverse_cache_key in document_comparisons:
            result = document_comparisons[reverse_cache_key]
            return {
                "similarity_score": result["similarity_score"],
                "common_topics": result["common_topics"],
                "unique_topics1": result["unique_topics2"],
                "unique_topics2": result["unique_topics1"],
                "document1_name": result["document2_name"],
                "document2_name": result["document1_name"]
            }
        
        text1 = await get_document_text(session_id1)
        text2 = await get_document_text(session_id2)
        doc1_name = f"Document {session_id1[:8]}"
        doc2_name = f"Document {session_id2[:8]}"
        
        if comparison_type == "semantic":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            chunks1 = [c for c in text1.split("\n\n") if len(c) > 100]
            chunks2 = [c for c in text2.split("\n\n") if len(c) > 100]
            
            max_chunks = 50
            if len(chunks1) > max_chunks:
                chunks1 = chunks1[::len(chunks1) // max_chunks]
            if len(chunks2) > max_chunks:
                chunks2 = chunks2[::len(chunks2) // max_chunks]
            
            embeddings1 = embeddings.embed_documents(chunks1)
            embeddings2 = embeddings.embed_documents(chunks2)
            
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            doc1_embedding = np.mean(embeddings1, axis=0)
            doc2_embedding = np.mean(embeddings2, axis=0)
            similarity = cosine_similarity([doc1_embedding], [doc2_embedding])[0][0]
            
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
            extract_topics_prompt = """
            Extract the main topics from the following document excerpt.
            Return a list of 5-10 topics as a JSON array of strings.
            Only return the JSON array, no additional text.
            
            Document:
            {text}
            """
            topics1_response = llm.invoke(extract_topics_prompt.format(text=text1[:10000]))
            topics2_response = llm.invoke(extract_topics_prompt.format(text=text2[:10000]))
            
            topics1_content = topics1_response.content.strip()
            topics2_content = topics2_response.content.strip()
            for prefix in ("```json", "```"):
                if topics1_content.startswith(prefix):
                    topics1_content = topics1_content[len(prefix):]
                if topics2_content.startswith(prefix):
                    topics2_content = topics2_content[len(prefix):]
            if topics1_content.endswith("```"):
                topics1_content = topics1_content[:-3]
            if topics2_content.endswith("```"):
                topics2_content = topics2_content[:-3]
            
            topics1 = json.loads(topics1_content)
            topics2 = json.loads(topics2_content)
            topics1_set = set(topics1)
            topics2_set = set(topics2)
            common_topics = list(topics1_set.intersection(topics2_set))
            unique_topics1 = list(topics1_set - topics2_set)
            unique_topics2 = list(topics2_set - topics1_set)
        else:
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            common_words = words1.intersection(words2)
            all_words = words1.union(words2)
            similarity = len(common_words) / len(all_words) if all_words else 0
            
            key_phrases1 = extract_key_phrases(text1, num_phrases=20)
            key_phrases2 = extract_key_phrases(text2, num_phrases=20)
            key_phrases1_set = set(key_phrases1)
            key_phrases2_set = set(key_phrases2)
            common_topics = list(key_phrases1_set.intersection(key_phrases2_set))
            unique_topics1 = list(key_phrases1_set - key_phrases2_set)
            unique_topics2 = list(key_phrases2_set - key_phrases1_set)
        
        result = {
            "similarity_score": float(similarity),
            "common_topics": common_topics,
            "unique_topics1": unique_topics1,
            "unique_topics2": unique_topics2,
            "document1_name": doc1_name,
            "document2_name": doc2_name
        }
        document_comparisons[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}")
        return {"error": f"Error comparing documents: {str(e)}"}

async def generate_visualization(session_id: str):
    try:
        if session_id in document_visualizations:
            return document_visualizations[session_id]
        
        text = await get_document_text(session_id)
        entities = extract_entities(text)
        key_phrases = extract_key_phrases(text, num_phrases=15)
        word_frequencies = get_word_frequencies(text, max_words=50)
        
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from sklearn.cluster import KMeans
        import numpy as np
        all_phrases = extract_key_phrases(text, num_phrases=100)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        phrase_embeddings = embeddings.embed_documents(all_phrases)
        
        num_clusters = min(8, len(all_phrases) // 5) if len(all_phrases) > 10 else 3
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(phrase_embeddings)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        clusters = {}
        for i, label in enumerate(labels):
            clusters.setdefault(label, []).append(all_phrases[i])
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(cluster_centers)
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        x_range = x_max - x_min or 1
        y_range = y_max - y_min or 1
        normalized_coords = [[(x - x_min) / x_range, (y - y_min) / y_range] for x, y in coords]
        
        topics = [
            {
                "id": f"topic_{i}",
                "label": clusters[label][0],
                "x": float(normalized_coords[i][0]),
                "y": float(normalized_coords[i][1]),
                "size": float(len(clusters[label]) / len(all_phrases)) * 2 + 0.5,
                "phrases": clusters[label][:3]
            }
            for i, label in enumerate(clusters.keys())
        ]
        
        visualization = {
            "topics": topics,
            "entities": entities,
            "key_phrases": key_phrases,
            "word_frequencies": word_frequencies
        }
        document_visualizations[session_id] = visualization
        return visualization
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return {"error": f"Error generating visualization: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

#uvicorn app:app --reload