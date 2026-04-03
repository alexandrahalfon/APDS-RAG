from typing import List, Dict
import numpy as np
from mistralai import Mistral
from query_processing import retrieve_context

def classify_query(query: str) -> str:
    '''Classify query type for adaptive processing'''
    q = query.lower()
    if any(word in q for word in ['what is', 'define', 'who is', 'when did']):
        return 'factual'
    elif any(word in q for word in ['how to', 'steps', 'process', 'procedure']):
        return 'procedural'
    elif any(word in q for word in ['compare', 'vs', 'difference', 'versus']):
        return 'comparative'
    elif any(word in q for word in ['why', 'explain', 'analyze', 'reason']):
        return 'analytical'
    elif any(word in q for word in ['list', 'enumerate', 'all', 'types of']):
        return 'listing'
    return 'general'

def handle_conversational_query(query: str) -> str:
    '''Handle basic conversational queries without document search'''
    q = query.lower().strip()
    
    # Greetings
    if any(greeting in q for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
        return "Hello! I'm a document analysis assistant. I can help you find information from uploaded PDF documents. What would you like to know about your documents?"
    
    # Goodbyes
    if any(bye in q for bye in ['goodbye', 'bye', 'see you later']):
        return "Goodbye! Feel free to ask me questions about your documents anytime."
    
    # Thanks
    if any(thanks in q for thanks in ['thank you', 'thanks']):
        return "You're welcome! Is there anything else you'd like to know about your documents?"
    
    # System questions
    if any(sys_q in q for sys_q in ['what can you do', 'what are your capabilities', 'help me']):
        return "I can help you analyze and extract information from uploaded PDF documents. You can ask me questions like:\n• What are the main findings?\n• Summarize the key points\n• What methodology was used?\n• What are the conclusions?\n\nJust ask me anything about your documents!"
    
    if any(name_q in q for name_q in ['what is your name', 'who are you']):
        return "I'm a RAG (Retrieval-Augmented Generation) assistant designed to help you analyze and query PDF documents. I can search through your uploaded documents and provide answers based on their content."
    
    if any(how_q in q for how_q in ['how does this work', 'how do i use this']):
        return "Here's how to use me:\n1. Upload PDF documents using the upload feature\n2. Ask me questions about the content\n3. I'll search through the documents and provide answers with source citations\n\nTry asking specific questions about your documents!"
    
    # Default conversational response
    return "I'm here to help you analyze your uploaded documents. Please ask me specific questions about the content of your PDFs, and I'll search through them to provide relevant answers."


def get_format_instruction(query_type: str) -> str:
    '''Get formatting instruction based on query type'''
    formats = {
        'factual': "Provide a clear definition followed by key characteristics.",
        'procedural': "Format your answer as step-by-step instructions with numbered steps.",
        'comparative': "Format your answer as a comparison with clear distinctions between items.",
        'analytical': "Explain the reasoning and underlying causes clearly.",
        'listing': "Format your answer as a numbered list with clear bullet points.",
        'general': "Provide a clear, concise answer."
    }
    return formats[query_type]


def check_entailment(answer, context_chunks, mistral_client):
    context = ' '.join([chunk['text'] for chunk in context_chunks])
    prompt = f"Context: {context}\nStatement: {answer}\nIs this statement supported by the context? YES/NO:"
    
    response = mistral_client.chat.complete(model='mistral-small', messages=[{'role': 'user', 'content': prompt}])
    return 'YES' in response.choices[0].message.content.upper()


def verify_answer(answer: str, context_chunks: List[Dict], mistral_client) -> str:
    '''Filter unsupported claims using entailment check only'''
    # Check overall entailment
    if not check_entailment(answer, context_chunks, mistral_client):
        # Provide relevant context summary as fallback
        context_text = ' '.join([chunk['text'] for chunk in context_chunks[:3]])  # Top 3 chunks
        summary_prompt = f"Summarize the key information from this context in 2-3 sentences: {context_text}"
        
        summary_response = mistral_client.chat.complete(
            model='mistral-small',
            messages=[{'role': 'user', 'content': summary_prompt}]
        )
        
        context_summary = summary_response.choices[0].message.content
        return f"Insufficient evidence: I cannot fully answer your question based on the provided context. However, here's what I found: {context_summary}"
    
    return answer


def check_query_safety(query: str) -> tuple[bool, str]:
    '''Check query safety and return (allowed, disclaimer)'''
    q = query.lower()
    
    # PII refusal
    if any(term in q for term in ['ssn', 'social security', 'phone number', 'address', 'email']):
        return False, "I cannot provide personal information from documents."
    
    # Medical disclaimer
    if any(term in q for term in ['diagnose', 'treatment', 'medication', 'symptoms', 'disease']):
        return True, "Medical disclaimer: This is informational only, not medical advice. Consult a healthcare professional."
    
    # Legal disclaimer
    if any(term in q for term in ['legal advice', 'lawsuit', 'contract', 'liability']):
        return True, "Legal disclaimer: This is informational only, not legal advice. Consult an attorney."
    
    return True, ""


def generate_answer_with_context(
    query: str,
    retrieved_chunks: List[Dict],
    mistral_client: Mistral,
    model: str = 'mistral-large-latest',
    min_threshold: float = 0.3
) -> str:
    '''Generate response using retrieved chunks as context'''
    # Check query safety
    allowed, disclaimer = check_query_safety(query)
    if not allowed:
        return disclaimer
    
    # Check evidence quality
    if not retrieved_chunks or retrieved_chunks[0]['similarity_score'] < min_threshold:
        return "Insufficient evidence: The retrieved information doesn't sufficiently match your query. Please try rephrasing your question or check if the information exists in the uploaded documents."
    
    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f'## Source {i} (Page {chunk["page"]})\n{chunk["text"]}'
        )
    
    context = '\n\n---\n\n'.join(context_parts)
    query_type = classify_query(query)
    format_instruction = get_format_instruction(query_type)
    
    # Create system and user prompts
    system_prompt = "You are a precise document analysis assistant. Extract information from provided documents to answer questions accurately without adding external knowledge. For basic greetings or conversational queries unrelated to the documents, politely redirect users to ask document-related questions."
    
    user_prompt = f"""# Document Context
{context}

# Question
{query}

# Instructions
- Answer based exclusively on the provided context
- If insufficient information, state "The provided documents do not contain enough information"
- For greetings or general conversation, politely ask the user to ask questions about the uploaded documents
- When citing information, reference the document title and actual page number from the source (e.g., "Page 3" not "Source 1")
- {format_instruction}
- Be concise and accurate

# Answer"""
    
    print('\nGenerating response...')
    
    # Generate response
    response = mistral_client.chat.complete(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        temperature=0.1,  # Low temperature for factual accuracy
        max_tokens=600
    )
    
    answer = response.choices[0].message.content
    
    # Verify answer against context
    verified_answer = verify_answer(answer, retrieved_chunks, mistral_client)
    
    # Add disclaimer if needed
    if disclaimer:
        verified_answer = f"{verified_answer}\n\n{disclaimer}"
    
    return verified_answer


def ask_document(
    query: str,
    embeddings: np.ndarray,
    metadata: List[Dict],
    mistral_client: Mistral,
    top_k: int = 5,
    show_sources: bool = True
) -> Dict:
    '''Ask question about document'''
    print('='*60)
    
    # Check for basic greetings/conversation - skip knowledge base search
    q = query.lower().strip()
    # Only trigger for very short, simple greetings
    if len(q.split()) <= 3 and any(q.startswith(greeting) for greeting in ['hello', 'hi', 'hey', 'good morning', 'thanks', 'thank you']):
        answer = f"Hello! I'm here to help you with questions about your uploaded documents. What would you like to know about them?"
        return {
            'query': query,
            'answer': answer,
            'sources': []
        }
    
    # Retrieve relevant context
    retrieved_chunks = retrieve_context(query, embeddings, metadata, mistral_client, top_k)
    
    # Generate answer
    answer = generate_answer_with_context(query, retrieved_chunks, mistral_client)
    
    print('='*60)
    print('\nResponse:')
    print(answer)
    
    if show_sources:
        print('\nSources:')
        for i, chunk in enumerate(retrieved_chunks, 1):
            paragraph_info = f", Paragraph {chunk['paragraph']}" if 'paragraph' in chunk else ""
            print(f'\n  Source {i} (Page {chunk["page"]}{paragraph_info}, Score: {chunk["similarity_score"]:.3f}):')
            print(f'  {chunk["text"][:150]}...')
    
    print('\n' + '='*60)
    
    return {
        'query': query,
        'answer': answer,
        'sources': retrieved_chunks
    }