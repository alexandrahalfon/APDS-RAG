import json
from typing import List, Dict
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def semantic_chunk_document(
    json_path: str, 
    chunk_size: int = 400,
    overlap_size: int = 50,
    output_dir: str = './chunks',
    save_to_disk: bool = True
) -> List[Dict]:
    '''Semantic chunking with sentence boundaries and overlap'''
    with open(json_path, 'r', encoding='utf-8') as f:
        structured_doc = json.load(f)
    
    chunks = []
    chunk_id = 0
    
    for page in structured_doc['pages']:
        # Get all sentences from page
        all_sentences = []
        for para in page['paragraphs']:
            sentences = sent_tokenize(para['text'])
            all_sentences.extend(sentences)
        
        # Create chunks respecting sentence boundaries
        current_chunk = {'id': chunk_id, 'page': page['page_number'], 'text': '', 'word_count': 0}
        
        for sentence in all_sentences:
            sentence_words = len(sentence.split())
            
            # If adding sentence exceeds chunk size, finalize current chunk
            if current_chunk['word_count'] + sentence_words > chunk_size and current_chunk['text']:
                chunks.append(current_chunk)
                chunk_id += 1
                
                # Start new chunk with overlap from previous
                overlap_text = ' '.join(current_chunk['text'].split()[-overlap_size:])
                current_chunk = {
                    'id': chunk_id,
                    'page': page['page_number'],
                    'text': overlap_text + ' ' if overlap_text else '',
                    'word_count': len(overlap_text.split()) if overlap_text else 0
                }
            
            current_chunk['text'] += sentence + ' '
            current_chunk['word_count'] += sentence_words
        
        if current_chunk['text']:
            chunks.append(current_chunk)
            chunk_id += 1
    
    # Clean up chunks
    for chunk in chunks:
        chunk['text'] = chunk['text'].strip()
        chunk['word_count'] = len(chunk['text'].split())
    
    if save_to_disk:
        chunks_dir = Path(output_dir)
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        with open(chunks_dir / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump({'chunks': chunks}, f, indent=2, ensure_ascii=False)
        
        print(f'✓ Saved {len(chunks)} semantic chunks')
    
    return chunks


