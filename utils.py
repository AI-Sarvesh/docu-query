import re
from collections import Counter
from typing import List, Dict, Any
import json
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, logger
import os

def count_syllables(word: str) -> int:
    """Count syllables in a word (approximate)."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    
    # Remove common endings
    if word.endswith('es') or word.endswith('ed'):
        word = word[:-2]
    elif word.endswith('e'):
        word = word[:-1]
    
    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    
    return max(1, count)

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract entities from text using regex patterns."""
    entities = {
        "people": [],
        "organizations": [],
        "locations": [],
        "dates": []
    }
    
    # Extract potential people (capitalized names)
    people_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    people_matches = re.findall(people_pattern, text)
    entities["people"] = list(set(people_matches))[:20]
    
    # Extract potential organizations
    org_pattern = r'\b[A-Z][A-Za-z]*([ ]?(Inc|Corp|LLC|Ltd|Company|Association|Organization))\b|\b[A-Z]{2,}\b'
    org_matches = re.findall(org_pattern, text)
    entities["organizations"] = list(set([match[0] if isinstance(match, tuple) else match for match in org_matches]))[:20]
    
    # Extract potential locations
    loc_pattern = r'\b[A-Z][a-z]+([ ][A-Z][a-z]+)*\b'
    loc_matches = re.findall(loc_pattern, text)
    locations = [loc for loc in loc_matches if loc not in entities["people"]]
    entities["locations"] = list(set(locations))[:20]
    
    # Extract dates
    date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'
    date_matches = re.findall(date_pattern, text)
    entities["dates"] = list(set([match if isinstance(match, str) else match[0] for match in date_matches]))[:20]
    
    return entities

def extract_key_phrases(text: str, num_phrases: int = 20) -> List[str]:
    """Extract key phrases using frequency analysis."""
    stop_words = set(['the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'or', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'])
    
    phrase_pattern = r'\b(?:[A-Za-z][a-z]+)(?: [A-Za-z][a-z]+){1,2}\b'
    phrases = re.findall(phrase_pattern, text.lower())
    
    filtered_phrases = []
    for phrase in phrases:
        words = phrase.split()
        if words[0] not in stop_words and words[-1] not in stop_words:
            filtered_phrases.append(phrase)
    
    phrase_counter = Counter(filtered_phrases)
    return [phrase for phrase, count in phrase_counter.most_common(num_phrases)]

def get_word_frequencies(text: str, max_words: int = 100) -> Dict[str, int]:
    """Get word frequencies from text."""
    words = re.findall(r'\b[A-Za-z][a-z]{2,}\b', text.lower())
    
    stop_words = set(['the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'or', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'should', 'now'])
    filtered_words = [word for word in words if word not in stop_words]
    
    word_counter = Counter(filtered_words)
    return dict(word_counter.most_common(max_words))

async def expand_query(query: str) -> List[str]:
    """
    Generate alternative phrasings for a query using a language model.
    
    Args:
        query: The original query
        
    Returns:
        A list of the original query plus alternative phrasings
    """
    try:
        llm = ChatOpenAI(temperature=0.7, model_name="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""
        I want you to generate 3 alternative phrasings of the following question. 
        Provide different ways to ask the same question that might help in retrieving relevant information.
        Return ONLY a JSON array of strings with the alternative phrasings. 
        
        Original question: {query}
        """
        
        response = await llm.invoke(prompt)
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Try to parse as JSON
        alt_queries = []
        try:
            # Handle case where response has code blocks
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip()
            else:
                json_str = response_content.strip()
                
            alt_queries = json.loads(json_str)
        except Exception as e:
            # If JSON parsing fails, try simple line splitting
            alt_queries = [q.strip() for q in response_content.split('\n') if q.strip() and not q.startswith('[') and not q.startswith(']')]
        
        # Ensure alt_queries is a list
        if not isinstance(alt_queries, list):
            alt_queries = [alt_queries]
        
        # Filter out any non-string elements
        alt_queries = [q for q in alt_queries if isinstance(q, str)]
        
        # Include original query and remove duplicates while maintaining order
        all_queries = [query] + alt_queries
        unique_queries = []
        seen = set()
        for q in all_queries:
            if q.lower() not in seen:
                unique_queries.append(q)
                seen.add(q.lower())
        
        return unique_queries
    except Exception as e:
        logger.error(f"Error generating query variations: {str(e)}")
        return [query]

async def generate_structured_query(query: str, doc_summary: str = "") -> Dict[str, Any]:
    """
    Generate a structured query for information retrieval.
    
    Args:
        query: The original user query
        doc_summary: A summary of the document for context
        
    Returns:
        A dictionary with structured query components
    """
    try:
        llm = ChatOpenAI(temperature=0.0, model_name="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
        
        prompt = f"""
        Based on the following user query and document summary, generate a structured retrieval query.
        Return ONLY a JSON object with the following fields:
        - main_topic: The primary topic or question
        - subtopics: List of related subtopics or aspects
        - keywords: List of important keywords to search for
        - excluded_terms: List of terms that might be ambiguous or lead to incorrect results
        
        User query: {query}
        Document summary: {doc_summary if doc_summary else 'Not available'}
        """
        
        response = await llm.invoke(prompt)
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Try to parse JSON response
        try:
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                json_str = response_content.split("```")[1].strip() 
            else:
                json_str = response_content.strip()
                
            structured_query = json.loads(json_str)
            
            # Ensure all required fields are present
            required_fields = ['main_topic', 'subtopics', 'keywords', 'excluded_terms']
            for field in required_fields:
                if field not in structured_query:
                    structured_query[field] = [] if field != 'main_topic' else query
                    
            return structured_query
        except Exception as e:
            logger.error(f"Error parsing structured query: {str(e)}")
            # Return a simple structured query based on the original query
            return {
                "main_topic": query,
                "subtopics": [],
                "keywords": query.split(),
                "excluded_terms": []
            }
    except Exception as e:
        logger.error(f"Error generating structured query: {str(e)}")
        return {
            "main_topic": query,
            "subtopics": [],
            "keywords": query.split(),
            "excluded_terms": []
        } 