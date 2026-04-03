import numpy as np
from typing import List, Dict
from mistralai import Mistral
from similarity_search import search_similar_chunks, merge_overlapping_chunks, rerank_by_diversity

from functools import lru_cache
import hashlib

# Cache for query embeddings
_query_cache = {}

def get_query_embedding(query: str, mistral_client: Mistral) -> np.ndarray:
    '''Convert user query to embedding with caching'''
    query_hash = hashlib.md5(query.encode()).hexdigest()
    
    if query_hash in _query_cache:
        return _query_cache[query_hash]
    
    response = mistral_client.embeddings.create(
        model='mistral-embed',
        inputs=[query]
    )
    embedding = np.array(response.data[0].embedding)
    _query_cache[query_hash] = embedding
    return embedding

def transform_query(query: str, mistral_client: Mistral) -> str:
    '''Transform query for better retrieval with caching'''
    query_hash = hashlib.md5(query.encode()).hexdigest()
    cache_key = f"transform_{query_hash}"
    
    if cache_key in _query_cache:
        return _query_cache[cache_key]
    
    prompt = f"Rewrite this search query to be more specific and include related terms: '{query}'"
    response = mistral_client.chat.complete(
        model='mistral-small',
        messages=[{'role': 'user', 'content': prompt}]
    )
    transformed = response.choices[0].message.content.strip()
    _query_cache[cache_key] = transformed
    return transformed


def retrieve_context(
    query: str,
    embeddings: np.ndarray,
    metadata: List[Dict],
    mistral_client: Mistral,
    top_k: int = 10
) -> List[Dict]:
    '''Retrieve most relevant chunks with adaptive weighting'''
    print(f"Query: '{query}'")
    
    # Transform and embed query
    enhanced_query = transform_query(query, mistral_client)
    query_embedding = get_query_embedding(enhanced_query, mistral_client)
    
    # Adaptive hybrid search based on query type
    from generation_step import classify_query
    query_type = classify_query(enhanced_query)
    alpha = {
        'factual': 0.8,      # Concepts matter more
        'procedural': 0.6,   # Steps need keywords
        'comparative': 0.7,  # Balanced approach
        'analytical': 0.7,   # Balanced approach
        'listing': 0.5,      # Keywords crucial
        'general': 0.7       # Default balanced
    }.get(query_type, 0.7)
    
    from similarity_search import hybrid_search
    results = hybrid_search(query_embedding, embeddings, metadata, enhanced_query, alpha=alpha, top_k=top_k * 2)
    
    # Post-process results
    merged_results = merge_overlapping_chunks(results)
    final_results = rerank_by_diversity(merged_results[:top_k])
    
    print(f'✓ Retrieved {len(final_results)} chunks (type: {query_type})')
    return final_results