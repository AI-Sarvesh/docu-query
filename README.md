# DocuQuery - PDF Document Q&A System

DocuQuery is an advanced document analysis and question-answering system that allows users to upload PDF documents, process them, and ask questions about their content. The system leverages NLP and LLM capabilities to provide accurate answers based on document content.

![DocuQuery Interface](static/screenshot.png)

## Features

- **PDF Document Processing**: Upload PDF documents for analysis and Q&A
- **Real-time Conversations**: Ask questions about document content with context-aware responses
- **Document Comparison**: Compare two documents to find similarities and differences
- **Document Visualization**: Generate visualizations of document topics, entities, and key phrases
- **Document Statistics**: Get detailed analytics on document readability, entities, and word frequencies
- **Feedback Collection**: Submit feedback on answers to improve the system
- **WebSocket Support**: Real-time updates during document processing and Q&A
- **Advanced NLP Options**: Enhanced text analysis with entity extraction and key phrase identification

## Architecture

DocuQuery is built on a modern, scalable architecture:

### Backend
- **FastAPI**: Asynchronous API framework with WebSocket support
- **LangChain**: Framework for LLM application development
- **FAISS**: Vector database for efficient similarity search
- **PyMuPDF**: PDF processing library
- **SQLite**: Lightweight database for feedback storage

### Frontend
- **HTML/CSS/JavaScript**: Responsive web interface
- **Tailwind CSS**: Utility-first CSS framework
- **D3.js**: Data visualization library

### Document Processing Pipeline
1. PDF Upload & Text Extraction
2. Document Structure Analysis
3. Semantic Chunking
4. Vector Embedding Generation
5. Hybrid Retrieval System (BM25 + Vector Search)
6. Context-Aware Q&A with OpenAI Models

## Setup and Installation

### Prerequisites

- Python 3.8+
- OpenAI API key
- Modern web browser

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/docu-query.git
   cd docu-query
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Run the application:
   ```
   uvicorn app:app --reload
   ```

6. Open your browser and go to http://localhost:8000

## Usage Guide

### Document Upload
1. Drag and drop a PDF file onto the upload area or click to select a file
2. Toggle "Advanced NLP processing" for more accurate results (slower processing)
3. Wait for document processing to complete

### Asking Questions
1. Type your question in the input field
2. View the system's response in the chat area
3. Provide feedback on answers with thumbs up/down buttons

### Document Comparison
1. Upload at least two documents
2. Select comparison type (Semantic or Text-based)
3. Choose a document to compare with
4. View similarity scores and topic analysis

### Document Insights
1. Click "Generate Document Insights" to analyze your document
2. Explore statistics, entities, and key phrases
3. View readability metrics and word frequencies

## Customization Options

### Configuration Settings
Edit `config.py` to customize:
- Server host and port
- Maximum sessions
- Processing options

### LLM Models
The system currently uses OpenAI's GPT models, but you can modify `services.py` to use alternative models.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgements

- OpenAI for providing the GPT models
- HuggingFace for embedding models
- FAISS for vector similarity search
- PyMuPDF for PDF processing
- FastAPI for the web framework 