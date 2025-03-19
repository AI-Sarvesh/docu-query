# DocuQuery - PDF Document Q&A System

DocuQuery is an advanced document analysis and question-answering system that allows users to upload PDF documents, process them, and ask questions about their content. The system leverages NLP and LLM capabilities to provide accurate answers based on document content.

## Features

- **PDF Document Processing**: Upload PDF documents for analysis and Q&A
- **Real-time Conversations**: Ask questions about document content with context-aware responses
- **Document Comparison**: Compare two documents to find similarities and differences
- **Document Visualization**: Generate visualizations of document topics, entities, and key phrases
- **Document Statistics**: Get detailed analytics on document readability, entities, and word frequencies
- **Feedback Collection**: Submit feedback on answers to improve the system
- **WebSocket Support**: Real-time updates during document processing and Q&A
- **Advanced NLP Options**: Enhanced text analysis with entity extraction and key phrase identification

## Technologies Used

- **Backend**: FastAPI, Python
- **NLP**: LangChain, OpenAI GPT models
- **Embeddings**: Hugging Face sentence transformers
- **Database**: SQLite for feedback storage
- **Frontend**: HTML, CSS, JavaScript (served via Jinja2 templates)
- **Document Processing**: PyMuPDF

## Setup and Installation

### Prerequisites

- Python 3.8+
- OpenAI API key

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

4. Create configuration file:
   Create a `config.py` file with the following content:
   ```python
   import os
   import logging

   # API Configuration
   HOST = "0.0.0.0"
   PORT = 8000

   # Directory configuration
   UPLOAD_DIR = "uploads"
   INDEXES_DIR = "indexes"

   # Create directories if they don't exist
   os.makedirs(UPLOAD_DIR, exist_ok=True)
   os.makedirs(INDEXES_DIR, exist_ok=True)

   # OpenAI API Key (replace with your actual key)
   OPENAI_API_KEY = "your-openai-api-key"

   # Configure logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   ```

5. Make sure to replace `"your-openai-api-key"` with your actual OpenAI API key.

### Running the Application

To run the application:

```bash
uvicorn app:app --reload
```

The application will be available at: http://localhost:8000

## Usage

1. Open your browser and navigate to http://localhost:8000
2. Upload a PDF document using the interface
3. Wait for the document to be processed (progress will be shown)
4. Ask questions about the document content
5. Explore document visualizations and statistics
6. Compare documents by uploading multiple PDFs

## API Endpoints

- `/` - Main interface
- `/upload/` - Upload PDF document
- `/status/{session_id}` - Check document processing status
- `/query/` - Query document content
- `/feedback/` - Submit feedback on answers
- `/summary/{session_id}` - Get document summary
- `/compare/` - Compare two documents
- `/visualization/{session_id}` - Get document visualization
- `/document_stats/{session_id}` - Get document statistics
- `/debug/{session_id}` - Toggle debug mode

## WebSocket API

Connect to `/ws/{session_id}` for real-time document processing and Q&A.

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 