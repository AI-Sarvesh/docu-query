import re
import spacy
import nltk
from typing import List, Dict, Any, Optional
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class FastChunker:
    """
    A faster alternative to AgenticChunker that uses local NLP models
    for document chunking with semantic awareness.
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 use_spacy: bool = False, spacy_model: str = "en_core_web_sm",
                 use_semantic_paragraphs: bool = True):
        """
        Initialize the FastChunker with configuration options.
        
        Args:
            chunk_size: Target size of chunks in characters
            chunk_overlap: Overlap between consecutive chunks in characters
            use_spacy: Whether to use spaCy for more advanced NLP (slower but better semantic chunking)
            spacy_model: The spaCy model to use if use_spacy is True
            use_semantic_paragraphs: Whether to try preserving paragraph/section boundaries
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_spacy = use_spacy
        self.use_semantic_paragraphs = use_semantic_paragraphs
        self.chunks = []
        
        # Initialize spaCy if needed
        if self.use_spacy:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"Downloading spaCy model {spacy_model}...")
                spacy.cli.download(spacy_model)
                self.nlp = spacy.load(spacy_model)
    
    def _extract_paragraphs(self, text: str) -> List[str]:
        """Extract paragraphs from text based on line breaks."""
        # Split by multiple newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK or spaCy."""
        if self.use_spacy:
            doc = self.nlp(text)
            return [sent.text for sent in doc.sents]
        else:
            return sent_tokenize(text)
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Process a document and return semantically meaningful chunks.
        
        Args:
            text: The document text to chunk
            
        Returns:
            A list of chunk strings
        """
        self.chunks = []
        
        if self.use_semantic_paragraphs:
            # Extract paragraphs first (preserve document structure)
            paragraphs = self._extract_paragraphs(text)
            
            current_chunk = ""
            for para in paragraphs:
                # If adding this paragraph exceeds chunk size and we already have content
                if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                    self.chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    sentences = self._split_into_sentences(current_chunk)
                    overlap_text = ""
                    overlap_length = 0
                    
                    # Build overlap from previous sentences
                    for sent in reversed(sentences):
                        if overlap_length + len(sent) <= self.chunk_overlap:
                            overlap_text = sent + " " + overlap_text
                            overlap_length += len(sent) + 1
                        else:
                            break
                    
                    current_chunk = overlap_text + para
                else:
                    # Add to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # Add the last chunk if it has content
            if current_chunk:
                self.chunks.append(current_chunk.strip())
        else:
            # Simple chunking by characters with overlap
            sentences = self._split_into_sentences(text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    self.chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
                else:
                    current_chunk += sentence + " "
            
            if current_chunk:
                self.chunks.append(current_chunk.strip())
        
        return self.chunks
    
    def get_chunks(self, get_type='list_of_strings'):
        """
        Return chunks in the requested format (compatible with AgenticChunker API).
        
        Args:
            get_type: The format to return chunks in ('list_of_strings' or 'dict')
            
        Returns:
            Chunks in the requested format
        """
        if get_type == 'list_of_strings':
            return self.chunks
        elif get_type == 'dict':
            return {
                f"chunk_{i}": {
                    'chunk_id': f"chunk_{i}",
                    'propositions': [chunk],
                    'title': f"Chunk {i+1}",
                    'summary': f"Content from document section {i+1}",
                    'chunk_index': i
                }
                for i, chunk in enumerate(self.chunks)
            }