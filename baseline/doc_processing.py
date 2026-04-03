import json
import re
import time
from typing import List, Dict
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize
from mistralai import Mistral
from mistralai.models import DocumentURLChunk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_processed_files(log_file_path):
    """Load list of already processed files from tracking file."""
    processed_files = set()
    if log_file_path.exists():
        with open(log_file_path, "r") as f:
            processed_files = set(line.strip() for line in f)
    return processed_files

def ocr_processing(pdf_path, mistral_client):
    '''Process PDF with mistral-ocr'''
    print(f"Processing: {pdf_path.name})")

    try:
        # Upload PDF to Mistral OCR
        uploaded_file = mistral_client.files.upload(
            file={
                'file_name': pdf_path.stem,
                'content': pdf_path.read_bytes(),
            },
            purpose='ocr',
        )

        # Get signed URL
        signed_url = mistral_client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

        # Perform OCR
        pdf_response = mistral_client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url),
            model='mistral-ocr-latest',
            include_image_base64=False
        )

        # Generate Markdown content
        markdown_output = ''
        for page in pdf_response.pages:
            page_number = page.index
            page_md = page.markdown
            if page_md:
                markdown_output += f'\n\n## Page {page_number}\n\n{page_md}\n'

        # Use first 2000 characters as context
        context_text = markdown_output[:2000]

        prompt = f'''
        You are a document metadata extraction specialist. 
        Analyze the OCR text and extract structured metadata for ANY type of document.

          INSTRUCTIONS:
          1. **Document Type**: First identify what type of document this is
          2. **Title**: Extract or create an appropriate title
          3. **Topic**: Provide a brief, relevant categorization
            - Be concise (2-6 words)
            - Be specific and practical with titles
            - Use 'Unable to determine' only if text is completely illegible
            - Do NOT make up information that isn't in the text

          OCR TEXT:
          {context_text}

          OUTPUT FORMAT:
          Respond ONLY with valid JSON in the following format (no other text):
          {{
              'document_type': 'type of document identified',
              'title': 'extracted or generated title here',
              'topic': 'concise category or topic here',
              'confidence': 'high|medium|low'
          }}'''

        response = mistral_client.chat.complete(
            model='mistral-large-latest',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            response_format={'type': 'json_object'}
        )

        # Parse the JSON response
        extracted_info = json.loads(response.choices[0].message.content)
        # Create the JSON structure with the new format
        result = {
            'file_name': pdf_path.name,
            'document_type': extracted_info.get("document_type", ""),
            'title': extracted_info.get("title", ""),
            'confidence': extracted_info.get("confidence", ""),
            'ocr_contents': markdown_output.strip(),
            'topic': extracted_info.get("topic", ""),
        }

        return result

    except Exception as e:
        print(f'Error extracting metadata: {str(e)}')
        return {
            'file_name': pdf_path.name,
            'document_type': 'unknown',
            'title': 'Untitled Document', 
            'topic': 'General',
            'confidence': 'error',
            'ocr_contents': ''
        }   

def split_by_pages(ocr_contents: str) -> Dict[int, str]:
    '''Split OCR content by pages'''
    pages = {}
    # Split by the ## Page N markers
    page_sections = re.split(r'## Page (\d+)', ocr_contents)
    
    # page_sections will be ['', '1', 'content1', '2', 'content2', ...]
    for i in range(1, len(page_sections), 2):
        page_num = int(page_sections[i])
        page_content = page_sections[i + 1].strip()
        pages[page_num] = page_content
    
    return pages

def split_paragraphs(text: str) -> List[str]:
    '''Split text into paragraphs'''
    text = re.sub(r'\n{3,}', '\n\n', text)
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip().replace('\n', ' ') for p in paragraphs]
    paragraphs = [p for p in paragraphs if len(p.split()) > 3]
    return paragraphs

def split_sentences(text: str) -> List[str]:
    '''Split text into sentences using NLTK'''
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]

def process_ocr_json(json_path: str):
    '''Process OCR JSON and create structured output'''
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    ocr_contents = data['ocr_contents']
    
    # Split by pages
    pages = split_by_pages(ocr_contents)
    
    # Process each page
    structured_data = {
        'file_name': data['file_name'],
        'title': data['title'],
        'topic': data['topic'],
        'pages': []
    }
    
    for page_num, page_text in sorted(pages.items()):
        page_paragraphs = split_paragraphs(page_text)
        
        page_data = {
            'page_number': page_num,
            'text': page_text,
            'paragraphs': []
        }
        
        for para_idx, para_text in enumerate(page_paragraphs):
            para_sentences = split_sentences(para_text)
            
            page_data['paragraphs'].append({
                'paragraph_number': para_idx + 1,
                'text': para_text,
                'sentences': para_sentences,
                'sentence_count': len(para_sentences),
                'word_count': len(para_text.split())
            })
        
        structured_data['pages'].append(page_data)
    
    return structured_data

def process_pdf_complete(pdf_path: str, mistral_client, output_dir: str = './structured_docs'):
    '''Complete PDF processing: OCR + Structure'''
    # Step 1: OCR processing
    ocr_result = ocr_processing(Path(pdf_path), mistral_client)
    
    # Step 2: Structure the OCR output
    structured_data = {
        'file_name': ocr_result['file_name'],
        'title': ocr_result['title'],
        'topic': ocr_result['topic'],
        'pages': []
    }
    
    # Split by pages and structure
    pages = split_by_pages(ocr_result['ocr_contents'])
    
    for page_num, page_text in sorted(pages.items()):
        page_paragraphs = split_paragraphs(page_text)
        
        page_data = {
            'page_number': page_num,
            'text': page_text,
            'paragraphs': []
        }
        
        for para_idx, para_text in enumerate(page_paragraphs):
            para_sentences = split_sentences(para_text)
            
            page_data['paragraphs'].append({
                'paragraph_number': para_idx + 1,
                'text': para_text,
                'sentences': para_sentences,
                'sentence_count': len(para_sentences),
                'word_count': len(para_text.split())
            })
        
        structured_data['pages'].append(page_data)
    
    # Save structured output
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / f"{Path(pdf_path).stem}_structured.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    
    return str(output_path)