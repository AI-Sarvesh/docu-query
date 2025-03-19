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

# Store document visualizations
document_visualizations = {}

# Store debug information
debug_info = {}

app = FastAPI(title="Document Q&A API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files
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

# Initialize database on startup
init_db()

@app.get("/")
async def read_root(request: Request):
    """Serve the main HTML page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_processing":
                temp_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_temp.pdf")
                # Get the advanced NLP option
                use_advanced_nlp = message.get("use_advanced_nlp", False)
                
                if os.path.exists(temp_file_path):
                    processing_task = asyncio.create_task(process_pdf(temp_file_path, session_id, websocket, use_advanced_nlp))
                    processing_tasks[session_id] = processing_task
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": "No document found to process."
                    }))
            
            elif message["type"] == "query":
                query = message["content"]
                
                if session_id not in conversation_chains:
                    chain_data = build_chain(session_id, websocket)
                    if not chain_data:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "content": "Session not found or document not processed yet."
                        }))
                        continue
                else:
                    chain_data = conversation_chains[session_id]
                
                memory = chain_data["memory"]
                memory.save_context(
                    {"input": query},
                    {"output": ""}
                )
                
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "Thinking..."
                }))
                
                await websocket.send_text(json.dumps({
                    "type": "start_stream"
                }))
                
                try:
                    streaming_chain = chain_data.get("streaming_chain")
                    
                    if not streaming_chain:
                        callback_handler = WebSocketCallbackHandler(websocket)
                        
                        streaming_llm = ChatOpenAI(
                            temperature=0, 
                            model_name="gpt-4o-mini", 
                            api_key=OPENAI_API_KEY,
                            streaming=True,
                            callbacks=[callback_handler]
                        )
                        
                        prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
                        retriever = chain_data.get("retriever")
                        
                        def format_docs(docs):
                            return "\n\n".join([f"Document section {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
                        
                        def get_history_aware_retriever(query):
                            chat_history = memory.load_memory_variables({}).get("chat_history", "")
                            augmented_query = f"Given the conversation history: {chat_history}\n\nAnswer this question: {query}"
                            docs = retriever.invoke(augmented_query)
                            
                            try:
                                reranked_docs = rerank_docs(query, docs, top_k=5)
                                return reranked_docs
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
                    
                    result = await streaming_chain.ainvoke(query)
                    
                    dont_know_phrases = [
                        "i don't know", "i do not know", "don't have enough information",
                        "cannot find", "no information", "not mentioned", "not provided",
                        "not specified", "not stated", "not found"
                    ]
                    
                    if any(phrase in result.lower() for phrase in dont_know_phrases):
                        logger.info(f"Initial response was 'I don't know'. Trying fallback retrieval for query: {query}")
                        
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
                            
                            fallback_llm = ChatOpenAI(
                                temperature=0.7, 
                                model_name="gpt-4o-mini", 
                                api_key=OPENAI_API_KEY
                            )
                            
                            fallback_response_obj = await fallback_llm.invoke(fallback_prompt)
                            fallback_response = fallback_response_obj.content if hasattr(fallback_response_obj, 'content') else str(fallback_response_obj)
                            
                            if not any(phrase in fallback_response.lower() for phrase in dont_know_phrases):
                                await websocket.send_text(json.dumps({
                                    "type": "token",
                                    "content": "\n\nLet me search more deeply in the document...\n\n"
                                }))
                                
                                await websocket.send_text(json.dumps({
                                    "type": "token",
                                    "content": fallback_response
                                }))
                                
                                result = result + "\n\nLet me search more deeply in the document...\n\n" + fallback_response
                        except Exception as e:
                            logger.error(f"Error in fallback retrieval: {str(e)}")
                    
                    memory.chat_memory.messages[-1].content = result
                    
                except Exception as e:
                    logger.error(f"Error with streaming response: {str(e)}")
                    
                    try:
                        regular_chain = chain_data.get("chain")
                        result = await regular_chain.ainvoke(query)
                        
                        await websocket.send_text(json.dumps({
                            "type": "token",
                            "content": result
                        }))
                        
                        memory.chat_memory.messages[-1].content = result
                    except Exception as e2:
                        error_message = f"Error processing query: {str(e2)}"
                        logger.error(error_message)
                        await websocket.send_text(json.dumps({
                            "type": "token",
                            "content": f"I'm sorry, I encountered an error while processing your query: {str(e2)}"
                        }))
                
                await websocket.send_text(json.dumps({
                    "type": "end_stream"
                }))
            
            elif message["type"] == "feedback":
                try:
                    conn = sqlite3.connect('feedback.db')
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO feedback (session_id, query, answer, rating) VALUES (?, ?, ?, ?)",
                        (session_id, message["query"], message["answer"], message["rating"])
                    )
                    conn.commit()
                    conn.close()
                    
                    await websocket.send_text(json.dumps({
                        "type": "feedback_received",
                        "content": "Thank you for your feedback!"
                    }))
                except Exception as e:
                    logger.error(f"Error saving feedback: {str(e)}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": f"Error saving feedback: {str(e)}"
                    }))
            
            elif message["type"] == "compare_documents":
                session_id1 = message.get("session_id1")
                session_id2 = message.get("session_id2")
                comparison_type = message.get("comparison_type", "semantic")
                
                if not session_id1 or not session_id2:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": "Both document session IDs are required for comparison."
                    }))
                    continue
                
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "Comparing documents..."
                }))
                
                result = await compare_documents(session_id1, session_id2, comparison_type)
                
                if "error" in result:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": result["error"]
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "comparison_result",
                        "content": result
                    }))
            
            elif message["type"] == "generate_visualization":
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "content": "Generating document visualization..."
                }))
                
                result = await generate_visualization(session_id)
                
                if "error" in result:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "content": result["error"]
                    }))
                else:
                    await websocket.send_text(json.dumps({
                        "type": "visualization_result",
                        "content": result
                    }))
    
    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "content": f"An error occurred: {str(e)}"
            }))
        except:
            pass

@app.post("/upload/", response_model=ProcessingStatus)
async def upload_document(file: UploadFile = File(...), use_advanced_nlp: bool = False):
    """Upload a PDF document and process it with PyMuPDF."""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    try:
        session_id = str(uuid.uuid4())
        temp_file_path = os.path.join(UPLOAD_DIR, f"{session_id}_temp.pdf")
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # For direct processing (not via WebSocket), start processing with the NLP option
        if not file.filename.startswith('_ws_'):  # Skip for WebSocket uploads
            processing_task = asyncio.create_task(process_pdf(temp_file_path, session_id, None, use_advanced_nlp))
            processing_tasks[session_id] = processing_task
        
        return ProcessingStatus(
            status="processing",
            message="Document uploaded. Connect via WebSocket for real-time updates.",
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
    else:
        raise HTTPException(status_code=404, detail="Summary not found for this session.")

@app.post("/feedback/", status_code=201)
async def submit_feedback(feedback: FeedbackRequest):
    try:
        conn = sqlite3.connect('feedback.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO feedback (session_id, query, answer, rating) VALUES (?, ?, ?, ?)",
            (feedback.session_id, feedback.query, feedback.answer, feedback.rating)
        )
        conn.commit()
        conn.close()
        return {"message": "Feedback submitted successfully"}
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving feedback: {str(e)}")

@app.post("/query/", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Legacy endpoint for non-WebSocket queries"""
    session_id = request.session_id
    
    if session_id not in conversation_chains:
        chain_data = build_chain(session_id)
        if not chain_data:
            raise HTTPException(status_code=404, detail="Session not found or document not processed yet.")
    else:
        chain_data = conversation_chains[session_id]
    
    chain = chain_data["chain"]
    memory = chain_data["memory"]
    
    memory.save_context(
        {"input": request.query},
        {"output": ""}
    )
    
    result = chain.invoke(request.query)
    memory.chat_memory.messages[-1].content = result
    
    return QueryResponse(answer=result, sources=[])

@app.post("/compare/", response_model=ComparisonResult)
async def compare_documents_endpoint(request: ComparisonRequest):
    result = await compare_documents(
        request.session_id1, 
        request.session_id2,
        request.comparison_type
    )
    
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
        sentence_count = len(re.findall(r'[.!?]+', text))
        paragraph_count = len(text.split('\n\n'))
        
        entities = extract_entities(text)
        key_phrases = extract_key_phrases(text, num_phrases=10)
        word_frequencies = dict(list(get_word_frequencies(text, max_words=20).items()))
        
        words = re.findall(r'\b\w+\b', text.lower())
        syllable_count = sum(count_syllables(word) for word in words)
        if sentence_count == 0:
            sentence_count = 1
        
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
        logger.exception("Full traceback:")
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

async def compare_documents(session_id1: str, session_id2: str, comparison_type: str = "semantic"):
    """Compare two documents and return similarity metrics."""
    try:
        # Check if we already have cached comparison results
        cache_key = f"{session_id1}_{session_id2}_{comparison_type}"
        reverse_cache_key = f"{session_id2}_{session_id1}_{comparison_type}"
        
        if cache_key in document_comparisons:
            return document_comparisons[cache_key]
        if reverse_cache_key in document_comparisons:
            result = document_comparisons[reverse_cache_key]
            # Swap document references for correct ordering
            return {
                "similarity_score": result["similarity_score"],
                "common_topics": result["common_topics"],
                "unique_topics1": result["unique_topics2"],
                "unique_topics2": result["unique_topics1"],
                "document1_name": result["document2_name"],
                "document2_name": result["document1_name"]
            }
        
        # Get document texts
        try:
            text1 = await get_document_text(session_id1)
            text2 = await get_document_text(session_id2)
        except FileNotFoundError as e:
            return {"error": str(e)}
        
        # Get document names (just use session IDs if real names not available)
        doc1_name = f"Document {session_id1[:8]}"
        doc2_name = f"Document {session_id2[:8]}"
        
        # For semantic comparison, use embeddings
        if comparison_type == "semantic":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Create chunk-level embeddings
            chunks1 = text1.split("\n\n")
            chunks2 = text2.split("\n\n")
            
            # Filter out very short chunks
            chunks1 = [c for c in chunks1 if len(c) > 100]
            chunks2 = [c for c in chunks2 if len(c) > 100]
            
            # Sample chunks if there are too many
            max_chunks = 50
            if len(chunks1) > max_chunks:
                step = len(chunks1) // max_chunks
                chunks1 = chunks1[::step]
            if len(chunks2) > max_chunks:
                step = len(chunks2) // max_chunks
                chunks2 = chunks2[::step]
            
            # Get embeddings
            embeddings1 = embeddings.embed_documents(chunks1)
            embeddings2 = embeddings.embed_documents(chunks2)
            
            # Calculate document-level similarity
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Average the chunk embeddings
            doc1_embedding = np.mean(embeddings1, axis=0)
            doc2_embedding = np.mean(embeddings2, axis=0)
            
            # Calculate cosine similarity
            similarity = cosine_similarity([doc1_embedding], [doc2_embedding])[0][0]
            
            # Identify topics in each document
            llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key=OPENAI_API_KEY)
            
            extract_topics_prompt = """
            Extract the main topics from the following document excerpt.
            Return a list of 5-10 topics as a JSON array of strings.
            Only return the JSON array, no additional text.
            
            Document:
            {text}
            """
            
            # Extract topics from document 1
            topics1_text = text1[:10000]  # Limit text length
            topics1_response = llm.invoke(extract_topics_prompt.format(text=topics1_text))
            
            # Extract topics from document 2
            topics2_text = text2[:10000]  # Limit text length
            topics2_response = llm.invoke(extract_topics_prompt.format(text=topics2_text))
            
            # Get the content from the response objects
            topics1_content = topics1_response.content if hasattr(topics1_response, 'content') else str(topics1_response)
            topics2_content = topics2_response.content if hasattr(topics2_response, 'content') else str(topics2_response)
            
            # Parse topics
            try:
                # Clean up responses to extract JSON
                topics1_content = topics1_content.strip()
                if topics1_content.startswith("```json"):
                    topics1_content = topics1_content[7:]
                if topics1_content.startswith("```"):
                    topics1_content = topics1_content[3:]
                if topics1_content.endswith("```"):
                    topics1_content = topics1_content[:-3]
                
                topics2_content = topics2_content.strip()
                if topics2_content.startswith("```json"):
                    topics2_content = topics2_content[7:]
                if topics2_content.startswith("```"):
                    topics2_content = topics2_content[3:]
                if topics2_content.endswith("```"):
                    topics2_content = topics2_content[:-3]
                
                topics1 = json.loads(topics1_content.strip())
                topics2 = json.loads(topics2_content.strip())
                
                # Find common and unique topics
                topics1_set = set(topics1)
                topics2_set = set(topics2)
                
                common_topics = list(topics1_set.intersection(topics2_set))
                unique_topics1 = list(topics1_set - topics2_set)
                unique_topics2 = list(topics2_set - topics1_set)
                
            except Exception as e:
                logger.error(f"Error parsing topics: {str(e)}")
                common_topics = []
                unique_topics1 = []
                unique_topics2 = []
        
        # For text comparison, use simpler metrics
        else:
            # Calculate Jaccard similarity at word level
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            common_words = words1.intersection(words2)
            all_words = words1.union(words2)
            
            similarity = len(common_words) / len(all_words) if all_words else 0
            
            # Extract key phrases from both documents
            key_phrases1 = extract_key_phrases(text1, num_phrases=20)
            key_phrases2 = extract_key_phrases(text2, num_phrases=20)
            
            # Find common and unique phrases (treating these as topics)
            key_phrases1_set = set(key_phrases1)
            key_phrases2_set = set(key_phrases2)
            
            common_topics = list(key_phrases1_set.intersection(key_phrases2_set))
            unique_topics1 = list(key_phrases1_set - key_phrases2_set)
            unique_topics2 = list(key_phrases2_set - key_phrases1_set)
        
        # Prepare result
        result = {
            "similarity_score": float(similarity),
            "common_topics": common_topics,
            "unique_topics1": unique_topics1,
            "unique_topics2": unique_topics2,
            "document1_name": doc1_name,
            "document2_name": doc2_name
        }
        
        # Cache the result
        document_comparisons[cache_key] = result
        
        return result
    
    except Exception as e:
        logger.error(f"Error comparing documents: {str(e)}")
        return {"error": f"Error comparing documents: {str(e)}"}

async def generate_visualization(session_id: str):
    """Generate visualization data for a document."""
    try:
        # Check if we already have cached visualization
        if session_id in document_visualizations:
            return document_visualizations[session_id]
        
        # Get document text
        try:
            text = await get_document_text(session_id)
        except FileNotFoundError:
            return {"error": "Document not found"}
        
        # Extract entities
        entities = extract_entities(text)
        
        # Extract key phrases
        key_phrases = extract_key_phrases(text, num_phrases=15)
        
        # Get word frequencies
        word_frequencies = get_word_frequencies(text, max_words=50)
        
        # Generate topics with cluster analysis
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from sklearn.cluster import KMeans
            import numpy as np
            
            # Get key phrases for topic modeling
            all_phrases = extract_key_phrases(text, num_phrases=100)
            
            # Get embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            phrase_embeddings = embeddings.embed_documents(all_phrases)
            
            # Determine number of clusters (topics)
            num_clusters = min(8, len(all_phrases) // 5) if len(all_phrases) > 10 else 3
            
            # Cluster the phrases
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            kmeans.fit(phrase_embeddings)
            
            # Get cluster centers
            cluster_centers = kmeans.cluster_centers_
            
            # Assign phrases to clusters
            labels = kmeans.labels_
            
            # Group phrases by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_phrases[i])
            
            # Calculate cluster center distances to get x, y coordinates for visualization
            from sklearn.decomposition import PCA
            
            # Reduce dimensions to 2D
            pca = PCA(n_components=2)
            coords = pca.fit_transform(cluster_centers)
            
            # Normalize coordinates to [0,1] range
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            if x_range == 0:
                x_range = 1
            if y_range == 0:
                y_range = 1
            
            normalized_coords = [
                [(x - x_min) / x_range, (y - y_min) / y_range]
                for x, y in coords
            ]
            
            # Create topic objects for visualization
            topics = []
            for i, label in enumerate(clusters.keys()):
                x, y = normalized_coords[i]
                
                # Calculate size based on number of phrases in cluster
                size = len(clusters[label]) / len(all_phrases)
                
                # Get representative phrases for the cluster
                representative_phrases = clusters[label][:3]
                
                topic = {
                    "id": f"topic_{i}",
                    "label": representative_phrases[0],
                    "x": float(x),
                    "y": float(y),
                    "size": float(size) * 2 + 0.5,  # Scale size for better visualization
                    "phrases": representative_phrases
                }
                
                topics.append(topic)
        
        except Exception as e:
            logger.error(f"Error generating topics: {str(e)}")
            topics = [
                {
                    "id": "fallback_topic",
                    "label": phrase,
                    "x": 0.1 + (i * 0.2) % 0.8,
                    "y": 0.1 + ((i * 0.3) % 0.8),
                    "size": 1.0,
                    "phrases": [phrase]
                }
                for i, phrase in enumerate(key_phrases[:5])
            ]
        
        # Create visualization data
        visualization = {
            "topics": topics,
            "entities": entities,
            "key_phrases": key_phrases,
            "word_frequencies": word_frequencies
        }
        
        # Cache the result
        document_visualizations[session_id] = visualization
        
        return visualization
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return {"error": f"Error generating visualization: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT) 

#uvicorn app:app --reload