import numpy as np
import json
from typing import List, Dict, Tuple
from mistralai import Mistral
from pathlib import Path
import math
from collections import Counter

def load_embeddings_from_files(embeddings_dir: str = './embeddings') -> Tuple[np.ndarray, List[Dict]]:
    '''Load embeddings and metadata from files'''
    embeddings_path = Path(embeddings_dir)
    
    # Load embeddings
    embeddings = np.load(embeddings_path / 'embeddings.npy')
    
    # Load metadata
    with open(embeddings_path / 'metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f'Loaded {len(embeddings)} embeddings from {embeddings_dir}/')
    
    return embeddings, metadata


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    '''Calculate cosine similarity between two vectors'''
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def search_similar_chunks(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    metadata: List[Dict],
    top_k: int = 10
) -> List[Dict]:
    '''Find top_k most similar chunks using cosine similarity'''
    similarities = []
    
    # Calculate similarity with each chunk
    for i, chunk_embedding in enumerate(embeddings):
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        similarities.append((i, similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top_k results
    results = []
    for idx, score in similarities[:top_k]:
        result = metadata[idx].copy()
        result['similarity_score'] = float(score)
        results.append(result)
    
    return results


def merge_overlapping_chunks(results: List[Dict], overlap_threshold: float = 0.8) -> List[Dict]:
    '''Merge chunks with high text overlap'''
    merged = []
    used = set()
    
    for i, chunk in enumerate(results):
        if i in used:
            continue
        
        merged_chunk = chunk.copy()
        for j, other in enumerate(results[i+1:], i+1):
            if j in used:
                continue
            
            # Simple overlap check by text similarity
            overlap = len(set(chunk['text'].split()) & set(other['text'].split())) / len(set(chunk['text'].split()) | set(other['text'].split()))
            
            if overlap > overlap_threshold:
                merged_chunk['text'] += ' ' + other['text']
                merged_chunk['similarity_score'] = max(merged_chunk['similarity_score'], other['similarity_score'])
                used.add(j)
        
        merged.append(merged_chunk)
        used.add(i)
    
    return merged


def rerank_by_diversity(results: List[Dict], diversity_weight: float = 0.3) -> List[Dict]:
    '''Rerank results to balance relevance and diversity'''
    if len(results) <= 1:
        return results
    
    reranked = [results[0]]  # Always include top result
    remaining = results[1:]
    
    while remaining:
        best_idx = 0
        best_score = -1
        
        for i, candidate in enumerate(remaining):
            # Calculate diversity penalty
            diversity_penalty = 0
            for selected in reranked:
                text_overlap = len(set(candidate['text'].split()) & set(selected['text'].split())) / len(set(candidate['text'].split()) | set(selected['text'].split()))
                diversity_penalty += text_overlap
            
            # Combined score
            combined_score = candidate['similarity_score'] - diversity_weight * diversity_penalty
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = i
        
        reranked.append(remaining.pop(best_idx))
    
    return reranked


def simple_bm25_score(query_terms, doc_terms, doc_len, avg_doc_len, doc_freq, total_docs, k1=1.5, b=0.75):
    '''Simple BM25 scoring without external libraries'''
    score = 0
    for term in query_terms:
        if term in doc_terms:
            tf = doc_terms[term]
            idf = max(0.1, math.log((total_docs - doc_freq.get(term, 0) + 0.5) / (doc_freq.get(term, 0) + 0.5)))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))
    return score


def keyword_search(query: str, documents: List[str], top_k: int = 10) -> List[tuple]:
    '''Simple BM25-style keyword search'''
    query_terms = query.lower().split()
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Calculate document frequencies
    doc_freq = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            doc_freq[term] += 1
    
    # Calculate average document length
    if not tokenized_docs:
        return []  # Return empty results if no documents
    avg_doc_len = sum(len(doc) for doc in tokenized_docs) / len(tokenized_docs)
    
    # Score each document
    scores = []
    for i, doc in enumerate(tokenized_docs):
        doc_terms = Counter(doc)
        score = simple_bm25_score(query_terms, doc_terms, len(doc), avg_doc_len, doc_freq, len(documents))
        scores.append((i, score))
    
    # Sort and return top results
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]

def hybrid_search(query_embedding, embeddings, metadata, query_text, alpha=0.7, top_k=10):
    '''Combine semantic and keyword search'''
    # Semantic scores
    semantic_results = search_similar_chunks(query_embedding, embeddings, metadata, 20)
    
    # Keyword scores  
    documents = [chunk['text'] for chunk in metadata]
    keyword_results = keyword_search(query_text, documents, 20)
    
    # Normalize BM25 scores to [0, 1] range
    if keyword_results:
        max_bm25_score = max(score for _, score in keyword_results)
        if max_bm25_score > 0:
            keyword_results = [(idx, score / max_bm25_score) for idx, score in keyword_results]
    
    # Create combined scores dictionary
    combined_scores = {}
    
    # Add semantic scores
    for result in semantic_results:
        idx = result.get('id', semantic_results.index(result))
        combined_scores[idx] = alpha * result['similarity_score']
    
    # Add normalized keyword scores
    for idx, score in keyword_results:
        if idx in combined_scores:
            combined_scores[idx] += (1 - alpha) * score
        else:
            combined_scores[idx] = (1 - alpha) * score
    
    # Sort by combined score and return top results
    sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    final_results = []
    for idx, score in sorted_indices:
        if idx < len(metadata):
            result = metadata[idx].copy()
            result['similarity_score'] = float(score)
            final_results.append(result)
    
    return final_results


def calculate_keyword_density(query: str, chunk_text: str) -> float:
    '''Calculate keyword density for better chunk scoring'''
    query_words = set(w.lower() for w in query.split() if len(w) > 2)
    chunk_words = chunk_text.lower().split()
    
    # Count query word occurrences
    matches = sum(1 for word in chunk_words if word in query_words)
    density = matches / (len(chunk_words) + 1)  # +1 for smoothing
    
    # Bonus for multiple different query words
    unique_matches = len(set(chunk_words) & query_words)
    diversity_bonus = unique_matches / len(query_words) if query_words else 0
    
    return density + (diversity_bonus * 0.3)

def enhanced_chunk_scoring(results: List[Dict], query: str) -> List[Dict]:
    '''Apply enhanced scoring to chunks'''
    for result in results:
        # Original similarity score
        base_score = result['similarity_score']
        
        # Keyword density boost
        keyword_density = calculate_keyword_density(query, result['text'])
        
        # Length penalty (very short/long chunks less useful)
        text_len = len(result['text'].split())
        if text_len < 20:
            length_penalty = 0.8
        elif text_len > 500:
            length_penalty = 0.9
        else:
            length_penalty = 1.0
        
        # Position boost (earlier content often more important)
        position_boost = max(0.8, 1.0 - (result.get('paragraph', 0) * 0.05))
        
        # Apply enhanced scoring
        enhanced_score = base_score * length_penalty * position_boost * (1 + keyword_density)
        result['similarity_score'] = enhanced_score
    
    return sorted(results, key=lambda x: x['similarity_score'], reverse=True)