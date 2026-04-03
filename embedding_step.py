import numpy as np
import json
import time
from typing import List, Dict, Tuple
from mistralai import Mistral
from pathlib import Path

def generate_embeddings(
    chunks: List[Dict], 
    mistral_client,
    batch_size: int = 3,
    delay_between_batches: float = 8.0,
    max_retries: int = 5,
    checkpoint_file: str = 'embedding_checkpoint.json',
    resume: bool = True,
    save_npy: bool = True,        
    output_dir: str = './embeddings',
    fast_mode: bool = True      
) -> List[Dict]:
    '''
    Generate embeddings with progress saving and resume capability
    
    Args:
        chunks: List of chunk dictionaries
        mistral_client: Mistral client instance
        batch_size: Number of chunks per batch
        delay_between_batches: Seconds to wait between batch requests
        max_retries: Maximum number of retries for failed requests
        checkpoint_file: File to save progress
        resume: Whether to resume from checkpoint
        save_npy: Whether to save in optimized format (npy + json)
        output_dir: Directory to save optimized embeddings
    '''
    # Fast mode settings for paid tier
    if fast_mode:
        original_batch_size = batch_size
        original_delay = delay_between_batches
        batch_size = 20  # Larger batches for paid tier
        delay_between_batches = 1.0  # Faster requests
    
    print(f'Generating embeddings for {len(chunks)} chunks...')
    print(f'Batch size: {batch_size}, Delay: {delay_between_batches}s')
    print(f'Rate: ~{60 / delay_between_batches:.1f} requests/min')
    
    # Try to resume from checkpoint
    start_idx = 0
    if resume and Path(checkpoint_file).exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('last_completed_index', 0)
            if start_idx > 0:
                print(f'Resuming from chunk {start_idx}...')
    
    # Split oversized chunks and extract texts
    processed_chunks = []
    for chunk in chunks:
        text = chunk['text']
        # Conservative token estimate: 1 word ≈ 1.5 tokens
        estimated_tokens = len(text.split()) * 1.5
        
        if estimated_tokens > 6000:  # Split if too large (conservative limit)
            words = text.split()
            chunk_size = 3000  # ~4500 tokens per sub-chunk (safe margin)
            for i in range(0, len(words), chunk_size):
                sub_chunk = chunk.copy()
                sub_chunk['text'] = ' '.join(words[i:i + chunk_size])
                processed_chunks.append(sub_chunk)
        else:
            processed_chunks.append(chunk)
    
    chunks = processed_chunks  # Replace original chunks
    texts = [chunk['text'] for chunk in chunks]
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(start_idx, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        
        # Retry logic with exponential backoff
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                # Call Mistral embedding API
                response = mistral_client.embeddings.create(
                    model='mistral-embed',
                    inputs=batch
                )
                
                # Extract embeddings
                batch_embeddings = [item.embedding for item in response.data]
                
                # Add embeddings to corresponding chunks immediately
                for j, embedding in enumerate(batch_embeddings):
                    chunks[i + j]['embedding'] = embedding
                
                success = True
                print(f'✓ Batch {batch_num}/{total_batches} | {min(i + batch_size, len(texts))}/{len(texts)} chunks')
                
                # Save checkpoint after each successful batch
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'last_completed_index': i + batch_size,
                        'total_chunks': len(texts),
                        'timestamp': time.time()
                    }, f)
                
                # Rate limiting: wait before next batch
                if i + batch_size < len(texts):
                    time.sleep(delay_between_batches)
                
            except Exception as e:
                error_str = str(e)
                
                # Check if it's a 429 or capacity error
                if '429' in error_str or 'capacity' in error_str.lower() or 'rate' in error_str.lower():
                    # If in fast mode and first rate limit hit, switch to conservative mode
                    if fast_mode and retry_count == 0:
                        print(f'⚠️  Rate limit hit - switching to conservative mode')
                        batch_size = original_batch_size
                        delay_between_batches = original_delay
                        fast_mode = False
                        print(f'   New settings: batch_size={batch_size}, delay={delay_between_batches}s')
                        # Restart this batch with new settings
                        continue
                    
                    retry_count += 1
                    
                    if retry_count < max_retries:
                        # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                        wait_time = 10 * (2 ** (retry_count - 1))
                        print(f'⚠️  Rate limit hit on batch {batch_num}/{total_batches}')
                        print(f'   Waiting {wait_time}s before retry {retry_count}/{max_retries}...')
                        time.sleep(wait_time)
                    else:
                        # Save progress before stopping
                        print(f'\nFailed batch {batch_num} after {max_retries} retries')
                        print(f'Progress saved. Completed up to chunk {i}')
                        print(f'To resume later, run the script again with resume=True')
                        
                        # Save partial results
                        partial_file = checkpoint_file.replace('.json', '_partial.json')
                        with open(partial_file, 'w', encoding='utf-8') as f:
                            json.dump(chunks, f, indent=2, ensure_ascii=False)
                        print(f'💾 Partial results saved to: {partial_file}')
                        
                        # Return what we have so far instead of crashing
                        return chunks
                else:
                    # Non-rate-limit error
                    print(f'Unexpected error on batch {batch_num}: {e}')
                    # Save progress
                    partial_file = checkpoint_file.replace('.json', '_partial.json')
                    with open(partial_file, 'w', encoding='utf-8') as f:
                        json.dump(chunks, f, indent=2, ensure_ascii=False)
                    raise
    
    # Clean up checkpoint file on successful completion
    if Path(checkpoint_file).exists():
        Path(checkpoint_file).unlink()
        print(f'Checkpoint file deleted (all done!)')
    
    print(f'✓ Completed all embeddings!')
    
    # Automatically save in .npy format
    if save_npy:
        print(f'\nSaving...')
        save_embeddings(chunks, output_dir)
    
    return chunks


def save_embeddings(chunks_with_embeddings: List[Dict], output_dir: str = './embeddings'):
    '''
    Save embeddings and metadata to separate files for fast loading
    '''
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract embeddings and metadata separately
    embeddings = []
    metadata = []
    
    for chunk in chunks_with_embeddings:
        if 'embedding' in chunk:
            embeddings.append(chunk['embedding'])
            metadata.append({
                'id': chunk['id'],
                'page': chunk['page'],
                'text': chunk['text'],
                'word_count': chunk['word_count'],
                'source_paragraphs': chunk.get('source_paragraphs', [])
            })
    
    # Save embeddings as numpy array (fast loading)
    embeddings_array = np.array(embeddings)
    np.save(output_path / 'embeddings.npy', embeddings_array)
    
    # Save metadata as JSON
    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f'Saved {len(embeddings)} embeddings to {output_dir}/')
    print(f'  - embeddings.npy: {embeddings_array.shape}')
    print(f'  - metadata.json: {len(metadata)} chunks')

