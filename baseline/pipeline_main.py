#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from mistralai import Mistral
from dotenv import load_dotenv

from doc_processing import process_pdf_complete
from chunking_step import semantic_chunk_document
from embedding_step import generate_embeddings, save_embeddings
from similarity_search import load_embeddings_from_files
from generation_step import ask_document

load_dotenv()

class RAGPipeline:
    def __init__(self):
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found")
        self.client = Mistral(api_key=api_key)
        self.embeddings = None
        self.metadata = None
        
    def process_documents(self, pdf_paths):
        all_chunks = []
        for pdf_path in pdf_paths:
            structured_json = process_pdf_complete(pdf_path, self.client)
            chunks = semantic_chunk_document(structured_json, save_to_disk=False)
            for chunk in chunks:
                chunk['source_file'] = Path(pdf_path).name
            all_chunks.extend(chunks)
        
        chunks_with_embeddings = generate_embeddings(all_chunks, self.client, save_npy=False)
        save_embeddings(chunks_with_embeddings)
        self.embeddings, self.metadata = load_embeddings_from_files()
        return len(all_chunks)
    
    def load_embeddings(self):
        self.embeddings, self.metadata = load_embeddings_from_files()
        return len(self.metadata)
    
    def query(self, question, top_k=5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded")
        return ask_document(question, self.embeddings, self.metadata, self.client, top_k, show_sources=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['process', 'query', 'interactive'])
    parser.add_argument('--files', nargs='+')
    parser.add_argument('--question', '-q')
    
    args = parser.parse_args()
    pipeline = RAGPipeline()
    
    if args.command == 'process':
        count = pipeline.process_documents(args.files)
        print(f"Processed {count} chunks")
        
    elif args.command == 'query':
        pipeline.load_embeddings()
        result = pipeline.query(args.question)
        print(f"Answer: {result['answer']}")
        
    elif args.command == 'interactive':
        pipeline.load_embeddings()
        while True:
            question = input("Question: ").strip()
            if question.lower() in ['quit', 'exit']:
                break
            result = pipeline.query(question)
            print(f"Answer: {result['answer']}\n")

if __name__ == '__main__':
    main()