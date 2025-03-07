# utils/text_processing.py


"""Text processing utilities for legal text analysis."""

import re
from typing import List, Set

def extract_articles(text: str) -> List[str]:
    """Extract article references from text.
    
    Args:
        text: Input text
        
    Returns:
        List of normalized article references
    """
    article_patterns = [
        r"[Aa]rt[íi]culo\s+(\d+(\s*[a-z])?)",
        r"[Aa]rt\.\s*(\d+(\s*[a-z])?)"
    ]
    
    found_articles = []
    for pattern in article_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            article_num = match[0].strip()
            found_articles.append(normalize_article(article_num))
    
    return list(set(found_articles))

def normalize_article(article_ref: str) -> str:
    """Normalize article references to a standard format.
    
    Args:
        article_ref: Article reference (e.g., "1º", "1 bis")
        
    Returns:
        Normalized article reference
    """
    # Remove non-alphanumeric characters except spaces
    clean_ref = re.sub(r'[^\w\s]', '', article_ref)
    # Replace multiple spaces with a single space
    clean_ref = re.sub(r'\s+', ' ', clean_ref)
    return f"artículo_{clean_ref.strip().lower()}"

def extract_legal_concepts(text: str, concept_list: List[str] = None) -> List[str]:
    """Extract legal concepts from text.
    
    Args:
        text: Input text
        concept_list: Optional list of concepts to look for
        
    Returns:
        List of extracted legal concepts
    """
    if concept_list is None:
        concept_list = [
            "dolo", "culpa", "imprudencia", "negligencia", "impericia", 
            "tentativa", "consumación", "desistimiento", "legítima defensa",
            "estado de necesidad", "inimputabilidad", "error de tipo", 
            "error de prohibición", "concurso ideal", "concurso real", 
            "concurso aparente", "reincidencia"
        ]
    
    found_concepts = []
    text_lower = text.lower()
    
    # Look for concepts from the list
    for concept in concept_list:
        if concept in text_lower:
            found_concepts.append(concept)
            
    # Look for definitions
    definition_patterns = [
        r"se\s+entiende\s+por\s+([^\.]+)",
        r"se\s+define\s+como\s+([^\.]+)",
        r"es\s+la\s+([^\.]+)",
        r"constituye\s+([^\.]+)"
    ]
    
    for pattern in definition_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Extract the main noun (approximate)
            words = match.split()
            if words and len(words) > 0:
                candidate = words[0].strip()
                # Only add if it's a reasonable length for a concept
                if 3 <= len(candidate) <= 30:
                    found_concepts.append(candidate)
    
    return list(set(found_concepts))

def extract_penalties(text: str) -> List[str]:
    """Extract penalties from text.
    
    Args:
        text: Input text
        
    Returns:
        List of extracted penalties
    """
    penalty_patterns = [
        r"pena\s+de\s+([^\.\,;]+)",
        r"sanción\s+de\s+([^\.\,;]+)",
        r"prisión\s+de\s+([^\.\,;]+)",
        r"multa\s+de\s+([^\.\,;]+)",
        r"reclusión\s+de\s+([^\.\,;]+)",
        r"inhabilitación\s+([^\.\,;]+)"
    ]
    
    penalties = []
    text_lower = text.lower()
    
    for pattern in penalty_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            # Clean and normalize the penalty text
            clean_penalty = match.strip()
            penalties.append(f"pena_{clean_penalty}")
    
    return list(set(penalties))

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Basic sentence splitting that handles common abbreviations in legal text
    text = re.sub(r'(?<=[^A-Z])\. (?=[A-Z])', '.\n', text)
    text = re.sub(r'(?<=\.)\s(?=[A-Z])', '\n', text)
    text = re.sub(r';', ';\n', text)
    
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    return sentences