import argparse
import os
import shutil
from pathlib import Path
import uvicorn
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from similarity_search import load_embeddings_from_files
from generation_step import ask_document

def query_documents(question, mistral_client):
    embeddings, metadata = load_embeddings_from_files()
    result = ask_document(question, embeddings, metadata, mistral_client)
    print(f"Answer: {result['answer']}")

def clear_all_data():
    """Clear all data directories to start fresh"""
    directories = ['chunks', 'structured_docs', 'uploads', 'embeddings']
    
    print("Clearing all data directories...")
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"✓ Cleared {dir_name}/")
            except Exception as e:
                print(f"✗ Error clearing {dir_name}/: {e}")
        else:
            print(f"- {dir_name}/ does not exist")
    
    print("\nAll data cleared! You can now start fresh with new documents.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['query', 'serve', 'ui', 'clear'])
    parser.add_argument('question', nargs='?', help='Question for query command')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
    elif args.command == 'ui':
        os.system("streamlit run ui.py")
    elif args.command == 'clear':
        clear_all_data()
    elif args.command == 'query':
        api_key = os.getenv('MISTRAL_API_KEY')
        if not api_key:
            print("Error: Set MISTRAL_API_KEY environment variable")
            exit(1)
        mistral_client = Mistral(api_key=api_key)
        query_documents(args.question, mistral_client)

if __name__ == '__main__':
    main()