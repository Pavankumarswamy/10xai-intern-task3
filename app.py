import os
import pymupdf
import faiss
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import json
from flask import Flask, render_template, request, jsonify, Response, stream_with_context, send_from_directory
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor
import requests
import uuid
import seaborn as sns
import google.generativeai as genai



app = Flask(__name__)
try:
    sns.set_theme(style="whitegrid")
except:
    pass
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit for large PDFs
executor = ThreadPoolExecutor(max_workers=4)

# ================= CONFIG =================
UPLOAD_FOLDER = 'uploads'
VECTOR_DB_FOLDER = 'vector_db'
PROCESSED_FOLDER = 'processed'

for folder in [UPLOAD_FOLDER, VECTOR_DB_FOLDER, PROCESSED_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

import threading
chart_lock = threading.Lock()

# Global variables
vector_index = None
chunks = []
current_system_prompt = "You are an AI assistant. Answer based on the provided context."
chat_histories = {}  # session_id -> list of messages

# HF PERSISTENCE REMOVED



# Firebase Config
FIREBASE_URL = "https://locatepro-3ecca-default-rtdb.asia-southeast1.firebasedatabase.app/"

def save_to_firebase(session_id, history):
    try:
        url = f"{FIREBASE_URL}/chats/{session_id}.json"
        # The user wants "/chats : session_id : { json string of all chat data }"
        # We put the string representation as the value for the session_id key
        requests.put(url, json=json.dumps(history))
    except Exception as e:
        print(f"Firebase Error: {e}")

def rewrite_query(user_query, history):
    """
    Uses LLM to rewrite the query + history into a standalone search query.
    """
    if not history: return user_query
    
    # Keep last 3 turns for context
    context_str = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-6:]])
    
    prompt = f"""
    [TASK: REWRITE SEARCH QUERY]
    Based on the Chat History, rewrite the User's Follow-up Question to be a standalone search query.
    Replace pronouns (it, he, she, that) with specific names/entities from history.
    
    CHAT HISTORY:
    {context_str}
    
    FOLLOW-UP QUESTION:
    {user_query}
    
    OUTPUT (Return ONLY the rewritten query string):
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash") # Use fast model for utility
        response = model.generate_content(prompt)
        rewritten = response.text.strip()
        print(f"DEBUG: Rewrote '{user_query}' -> '{rewritten}'")
        return rewritten
    except:
        return user_query
# Suppress warnings & logs for clean streaming
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import sys
import gc

# Lazy load embedder to save memory at startup
embedder = None

def get_gemini_embeddings(texts, is_query=False):
    """
    Uses Google Gemini API to generate embeddings.
    Handles both single strings and lists of strings.
    """
    task_type = "retrieval_query" if is_query else "retrieval_document"
    try:
        # Gemini embedding model
        model_name = "models/embedding-001"
        
        if isinstance(texts, str):
            result = genai.embed_content(
                model=model_name,
                content=texts,
                task_type=task_type
            )
            return np.array(result['embedding']).astype("float32")
        else:
            # Batch processing for lists
            result = genai.embed_content(
                model=model_name,
                content=texts,
                task_type=task_type
            )
            return np.array(result['embedding']).astype("float32")
    except Exception as e:
        print(f"❌ Gemini Embedding Error: {e}")
        sys.stdout.flush()
        return None

# ================= CONFIG =================
# ================= CONFIG =================
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()


# Security: Load API Key from Environment
encrypted_key = os.getenv("GEMINI_API_KEY")

def decrypt_key(encrypted_key: str) -> str:
    if not encrypted_key: return ""
    decrypted = ""
    for ch in encrypted_key:
        if 'A' <= ch <= 'Z':
            decrypted += chr((ord(ch) - ord('A') - 3) % 26 + ord('A'))
        elif 'a' <= ch <= 'z':
            decrypted += chr((ord(ch) - ord('a') - 3) % 26 + ord('a'))
        elif '0' <= ch <= '9':
            decrypted += str((int(ch) - 1) % 10)
        else:
            decrypted += ch
    return decrypted

if not encrypted_key:
    print("⚠️ WARNING: GEMINI_API_KEY not found in environment variables.")
    GEMINI_API_KEY = None
else:
    # Check if the key looks like it needs decryption (doesn't start with AIza)
    if not encrypted_key.startswith("AIza"):
        print("🔐 Decrypting API Key...")
        GEMINI_API_KEY = decrypt_key(encrypted_key)
    else:
        GEMINI_API_KEY = encrypted_key

genai.configure(api_key=GEMINI_API_KEY)
CURRENT_MODEL = "gemini-2.0-flash"

# ================= PERSISTENCE =================

def save_vector_db():
    global vector_index, chunks
    if vector_index is None or not chunks:
        return
    
    try:
        # Save FAISS index
        index_path = os.path.join(VECTOR_DB_FOLDER, "faiss_index.bin")
        faiss.write_index(vector_index, index_path)
        
        # Save chunks metadata (compact format, no indent for speed)
        chunks_path = os.path.join(VECTOR_DB_FOLDER, "chunks.json")
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, separators=(',', ':'))
            
        # Generate vectors.json (compact format)
        vectors_data = [{
            "text": c['text'],
            "metadata": c['meta'],
            "embedding": c.get('embedding', [])
        } for c in chunks]
        
        vectors_path = os.path.join(VECTOR_DB_FOLDER, "vectors.json")
        with open(vectors_path, 'w', encoding='utf-8') as f:
            json.dump(vectors_data, f, separators=(',', ':'))
        
        print(f"💾 Saved {len(chunks)} chunks to DB")
        

                    
    except Exception as e:
        print(f"❌ Save Error: {e}")


def load_vector_db():
    global vector_index, chunks
    index_path = os.path.join(VECTOR_DB_FOLDER, "faiss_index.bin")
    chunks_path = os.path.join(VECTOR_DB_FOLDER, "chunks.json")
    
    if os.path.exists(index_path) and os.path.exists(chunks_path):
        try:
            print("📂 Loading existing vector database...")
            sys.stdout.flush()
            
            with open(chunks_path, 'r', encoding='utf-8') as f:
                raw_chunks = json.load(f)
            
            # 1. Deduplicate by Text + Meta
            seen = set()
            unique_chunks = []
            for c in raw_chunks:
                key = (c['text'].strip(), c.get('meta'))
                if key not in seen and len(c['text'].strip()) > 10:
                    unique_chunks.append(c)
                    seen.add(key)
            
            # 2. Check for missing embeddings and re-encode if necessary
            missing_emb = [i for i, c in enumerate(unique_chunks) if 'embedding' not in c or not c['embedding']]
            
            if missing_emb or len(unique_chunks) < len(raw_chunks):
                print(f"🧹 Reconciling DB: {len(raw_chunks)} -> {len(unique_chunks)} chunks.")
                sys.stdout.flush()
                
                # Get embedder (lazy load)
                model = get_embedder()
                
                # Re-encode EVERYTHING to ensure consistency if any are missing
                texts = [c['text'] for c in unique_chunks]
                embeddings = model.encode(texts, show_progress_bar=False, batch_size=8).astype("float32")
                
                vector_index = faiss.IndexFlatL2(embeddings.shape[1])
                vector_index.add(embeddings)
                
                # Update chunks with new embeddings
                emb_list = embeddings.tolist()
                for i, c in enumerate(unique_chunks):
                    c['embedding'] = emb_list[i]
                
                chunks = unique_chunks
                save_vector_db() # Sync cleaned version
            else:
                chunks = unique_chunks
                vector_index = faiss.read_index(index_path)

            print(f"✅ Loaded {len(chunks)} chunks from existing vector DB.")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"❌ Error loading vector DB: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            vector_index = None
            chunks = []
    else:
        print("ℹ️  No existing vector database found. Starting fresh.")
        print("   Upload documents to build the knowledge base.")
        sys.stdout.flush()
        vector_index = None
        chunks = []



def save_processed_data(segments):
    """
    Saves structured (tables) and unstructured (narrative) data separately.
    """
    structured = [s for s in segments if s.get('type') == 'table']
    unstructured = [s for s in segments if s.get('type') == 'narrative']
    
    # We append to existing files if they exist, or create new ones
    struct_path = os.path.join(PROCESSED_FOLDER, "structured.json")
    unstruct_path = os.path.join(PROCESSED_FOLDER, "unstructured.json")
    
    def append_to_json(path, new_data):
        existing = []
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except: pass
        
        # Intelligent deduplication based on text and source
        seen_keys = set((s['text'], s.get('source'), s.get('page')) for s in existing)
        added_count = 0
        
        for s in new_data:
            key = (s['text'], s.get('source'), s.get('page'))
            if key not in seen_keys:
                existing.append(s)
                seen_keys.add(key)
                added_count += 1
        
        if added_count > 0:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=4)
            print(f"   -> Added {added_count} unique segments to {os.path.basename(path)}")

    if structured: append_to_json(struct_path, structured)
    if unstructured: append_to_json(unstruct_path, unstructured)
    print(f"✅ Processed data saved to {PROCESSED_FOLDER}")
    sys.stdout.flush()



# Load DB on startup
print("="*50)
print("🚀 Initializing RAG Application...")
print("="*50)
sys.stdout.flush()

load_vector_db()

print("="*50)
print("✅ Application Ready!")
print("="*50)
sys.stdout.flush()

# ================= HELPERS: Extraction =================

# ================= HELPERS: Extraction & Chunking =================

# ================= HELPERS: Extraction & Chunking =================

def extract_content(file_path):
    """
    Returns a list of raw segments:
    [{'text': str, 'source': str, 'page': int/str}, ...]
    """
    filename = os.path.basename(file_path)
    ext = file_path.rsplit('.', 1)[-1].lower()
    segments = []
    
    try:
        if ext == 'pdf':
            doc = pymupdf.open(file_path)
            for i, page in enumerate(doc):
                text = page.get_text()
                if len(text.strip()) > 50:
                    segments.append({
                        'text': text, 
                        'source': filename,
                        'page': i + 1,
                        'type': 'narrative'
                    })
                
                # Check tables per page
                tabs = page.find_tables()
                if tabs.tables:
                    for j, tab in enumerate(tabs):
                        try:
                            df = tab.to_pandas()
                            md = df.to_markdown(index=False)
                            segments.append({
                                'text': md,
                                'source': filename,
                                'page': i + 1,
                                'title': f"Table {j+1}",
                                'type': 'table'
                            })
                        except: pass
                
        elif ext in ['xlsx', 'xls', 'csv']:
            # For excel/csv, treat sheets as "pages"
            if ext == 'csv':
                df = pd.read_csv(file_path)
                segments.append({'text': df.to_markdown(), 'source': filename, 'page': 'Sheet1', 'type': 'table'})
            else:
                xls = pd.ExcelFile(file_path)
                for sheet in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet)
                    segments.append({'text': df.to_markdown(), 'source': filename, 'page': sheet, 'type': 'table'})
                    
        elif ext in ['txt', 'json']:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                segments.append({'text': content, 'source': filename, 'page': 1, 'type': 'narrative'})
                
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        
    return segments



# ================= INTELLIGENCE: Gemini Implementation =================

def fast_paragraph_splitter(text, max_chars=1200):
    """
    Ultra-fast text chunking optimized for speed.
    Splits on double newlines first, then by sentences if needed.
    """
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        # If single paragraph is too large, split by sentences
        if len(p) > max_chars:
            # Split by common sentence endings
            sentences = [s.strip() + '.' for s in p.split('. ') if s.strip()]
            for sent in sentences:
                if len(current_chunk) + len(sent) < max_chars:
                    current_chunk += sent + " "
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sent + " "
        else:
            if len(current_chunk) + len(p) < max_chars:
                current_chunk += p + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Filter out very small chunks
    return [c for c in chunks if len(c) > 20]



def generate_chart(context, query):
    schema = """{"type": "bar|line|pie|scatter", "title": "...", "categories": ["Label 1", "Label 2"], "values": [10, 20]}"""
    prompt = f"""
    [TASK: EXTRACT CHART DATA]
    User Query: "{query}"
    
    Rules:
    1. Return valid JSON matching ONLY: {schema}
    2. 'categories': List of strings (X-axis labels).
    3. 'values': List of numbers (Y-axis values).
    4. LENGTHS MUST MATCH EXACTLY.
    5. LIMIT TO TOP 12 ITEMS ONLY to avoid visual clutter.
    6. If data is complex, FLATTEN it into labels (e.g., "Rohan (Math)").
    
    CONTEXT: {context[:4000]}
    """
    try:
        model = genai.GenerativeModel(CURRENT_MODEL)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        # Sanitize JSON (remove markdown fences if present)
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        data = json.loads(text.strip())
        
        # Plotting (Matplotlib logic same as before)
        with chart_lock:
            plt.figure(figsize=(8, 5))
            if data['type'] == 'bar':
                ax = sns.barplot(x=data['categories'], y=data['values'], hue=data['categories'], palette='magma', legend=False)
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.1f', padding=3)
            elif data['type'] == 'line':
                sns.lineplot(x=data['categories'], y=data['values'], marker='o')
            elif data['type'] == 'pie':
                plt.pie(data['values'], labels=data['categories'], autopct='%1.1f%%')
            elif data['type'] == 'scatter':
                sns.scatterplot(x=data['categories'], y=data['values'])
                
            plt.title(data.get('title', 'Chart'), fontsize=12)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        return img_base64
    except Exception as e:
        print(f"Chart Error: {e}")
        return None

def split_and_vectorize(segments, batch_size=8):
    """
    Memory-optimized vectorization for HuggingFace Spaces.
    Uses smaller batches and progressive processing.
    """
    global vector_index, chunks
    
    if not segments:
        return
    
    new_chunks = []
    
    # Fast chunking without LLM calls
    for seg in segments:
        text = seg['text'].strip()
        if len(text) < 20:  # Skip very short content
            continue
            
        meta = f"{seg['source']} (Page {seg['page']})"
        
        if seg['type'] == 'narrative':
            # Fast paragraph splitting
            sub_chunks = fast_paragraph_splitter(text)
            for sub in sub_chunks:
                new_chunks.append({'text': sub, 'meta': meta})
        else:
            # Tables: keep as-is if small, otherwise split
            if len(text) < 2000:
                new_chunks.append({
                    'text': f"### Table from {meta}\n{text}",
                    'meta': meta
                })
            else:
                # Fast line-based splitting for large tables
                lines = text.split('\n')
                chunk_str = ""
                for line in lines:
                    if len(chunk_str) + len(line) > 1200:
                        if chunk_str:
                            new_chunks.append({'text': chunk_str, 'meta': meta})
                        chunk_str = line + "\n"
                    else:
                        chunk_str += line + "\n"
                if chunk_str:
                    new_chunks.append({'text': chunk_str, 'meta': meta})

    if not new_chunks:
        return

    # Fast duplicate detection using hash set
    existing_hashes = set(hash((c['text'], c['meta'])) for c in chunks)
    unique_new_chunks = [
        c for c in new_chunks 
        if hash((c['text'], c['meta'])) not in existing_hashes
    ]
    
    if not unique_new_chunks:
        print("✓ All chunks already indexed")
        sys.stdout.flush()
        return

    # Batch encode using Gemini API (No local model needed!)
    texts = [c['text'] for c in unique_new_chunks]
    total = len(texts)
    print(f"⚡ Encoding {total} chunks via Gemini API...")
    sys.stdout.flush()
    
    # Process in batches of 50 (Gemini API limit is usually around 100 for batch embedding)
    all_embeddings = []
    batch_size = 50 
    for i in range(0, total, batch_size):
        batch = texts[i:i+batch_size]
        try:
            batch_emb = get_gemini_embeddings(batch)
            if batch_emb is not None:
                all_embeddings.append(batch_emb)
            
            # Progress feedback
            print(f"  Processed {min(i + batch_size, total)}/{total} chunks")
            sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ API Encoding error at batch {i}: {e}")
            sys.stdout.flush()
            continue
    
    if not all_embeddings:
        print("❌ No embeddings generated")
        sys.stdout.flush()
        return
    
    embeddings = np.vstack(all_embeddings).astype("float32")
    
    # Initialize or update index
    if vector_index is None:
        vector_index = faiss.IndexFlatL2(embeddings.shape[1])
        chunks = []
    
    vector_index.add(embeddings)
    
    # Store embeddings efficiently
    emb_list = embeddings.tolist()
    for i, c in enumerate(unique_new_chunks):
        c['embedding'] = emb_list[i]
    
    chunks.extend(unique_new_chunks)
    
    # Clean up memory
    del embeddings, all_embeddings
    gc.collect()
    
    # Background save
    executor.submit(save_vector_db)
    
    print(f"✅ Indexed {len(unique_new_chunks)} new chunks (Total: {len(chunks)})")
    sys.stdout.flush()

def retrieve_chunks(query, k=5):
    """
    Retrieves top-k relevant chunks from the vector database.
    """
    if vector_index is None or len(chunks) == 0:
        return []
    
    query_embedding = get_gemini_embeddings(query, is_query=True)
    if query_embedding is None: return []
    
    # Reshape for FAISS
    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = vector_index.search(query_embedding, k=k)
    
    retrieved_items = []
    for idx in indices[0]:
        if idx < len(chunks) and idx != -1:
            retrieved_items.append(chunks[idx])
    return retrieved_items




# ================= ROUTES =================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files: 
        return jsonify({"error": "No file"}), 400
    
    files = request.files.getlist('file')
    if not files: 
        return jsonify({"error": "No selection"}), 400
    
    processed_count = 0
    all_segments = []
    
    try:
        print(f"📤 Processing {len(files)} file(s)...")
        sys.stdout.flush()  # Force immediate output for HF Spaces
        
        for idx, file in enumerate(files, 1):
            if file.filename == '': 
                continue
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            print(f"  [{idx}/{len(files)}] Saving {filename}...")
            sys.stdout.flush()
            file.save(file_path)

            
            # Extract content
            print(f"  [{idx}/{len(files)}] Extracting {filename}...")
            sys.stdout.flush()
            
            try:
                segments = extract_content(file_path)
                
                if segments:
                    all_segments.extend(segments)
                    processed_count += 1
                    print(f"  ✓ Extracted {len(segments)} segments from {filename}")
                    sys.stdout.flush()
            except Exception as e:
                print(f"  ⚠️ Error extracting {filename}: {e}")
                sys.stdout.flush()
                continue
        
        # Batch vectorize all segments at once (much faster!)
        if all_segments:
            print(f"⚡ Batch indexing {len(all_segments)} segments...")
            sys.stdout.flush()
            
            try:
                split_and_vectorize(all_segments)
                
                # Save processed data
                save_processed_data(all_segments)
                
                print(f"✅ Upload complete!")
                sys.stdout.flush()
                
            except Exception as e:
                print(f"❌ Indexing Error: {e}")
                sys.stdout.flush()
                return jsonify({"error": f"Indexing failed: {str(e)}"}), 500
        
        return jsonify({
            "message": f"✅ Successfully indexed {processed_count} file(s)!",
            "system_prompt": "Knowledge base updated." 
        })
        
    except Exception as e:
        print(f"❌ Upload Error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({"error": str(e)}), 500

@app.route('/files', methods=['GET'])
def list_files():
    try:
        files = os.listdir(UPLOAD_FOLDER)
        return jsonify({"files": files})
    except:
        return jsonify({"files": []})

# Revert process_existing to use new logic (simplified)
@app.route('/process_existing', methods=['POST'])
def process_existing():
    try:
        data = request.json
        filename = data.get("filename")
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(file_path): 
            return jsonify({"error": "File missing"}), 404
        
        segments = extract_content(file_path)
        if not segments:
            return jsonify({"error": "Could not extract content from file"}), 400
            
        split_and_vectorize(segments)
        save_processed_data(segments)
        
        return jsonify({"message": f"Added {filename} to knowledge base."})
    except Exception as e:
        print(f"Process Existing Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/delete_file', methods=['DELETE'])
def delete_file():
    global vector_index, chunks
    data = request.json
    filename = data.get("filename")
    if not filename: return jsonify({"error": "No filename"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    
    # Remove from vector DB
    if chunks:
        # Rebuild index without the deleted file
        new_chunks = [c for c in chunks if filename not in c['meta']]
        if len(new_chunks) != len(chunks):
            chunks = new_chunks
            if chunks:
                texts = [c['text'] for c in chunks]
                # Optimized: Use existing embeddings instead of re-encoding
                embeddings = np.array([c['embedding'] for c in chunks]).astype("float32")
                vector_index = faiss.IndexFlatL2(embeddings.shape[1])
                vector_index.add(embeddings)
            else:
                vector_index = None
            save_vector_db()
            
    return jsonify({"message": f"Deleted {filename}"})

@app.route('/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_query = data.get("query", "")
    session_id = data.get("session_id", "default_session")
    
    # Initialize history if not present (try to load from Firebase if missing in memory)
    if session_id not in chat_histories:
        try:
            url = f"{FIREBASE_URL}/chats/{session_id}.json"
            res = requests.get(url, timeout=5)
            if res.status_code == 200 and res.json():
                chat_histories[session_id] = json.loads(res.json())
                print(f"✅ Loaded history for session {session_id} from Firebase.")
            else:
                chat_histories[session_id] = []
        except:
            chat_histories[session_id] = []
    
    history = chat_histories[session_id]
    
    if vector_index is None:
        return jsonify({"error": "Please select a file from the sidebar first."})
        
    # 1. REWRITE QUERY using History
    search_query = rewrite_query(user_query, history)
    
    # 2. Retrieve with Intent Detection
    data_keywords = ['all', 'list', 'every', 'average', 'summary', 'calculate', 'total', 'who are', 'names', 'students', 'table']
    is_data_query = any(k in user_query.lower() for k in data_keywords)
    
    # Use higher k for data queries to ensure we get ALL relevant rows/info
    k_value = 25 if is_data_query else 6
    context_chunks = retrieve_chunks(search_query, k=k_value)
    
    # Build text representation for context
    context = "\n".join([c['text'] for c in context_chunks])
    retrieved_context = "\n".join([f"[Source: {c['meta']}]\n{c['text']}" for c in context_chunks])

    # Expanded keywords to make chart generation "more important" and frequent
    chart_keywords = [
        'chart', 'graph', 'plot', 'visualize', 'trend', 
        'compare', 'comparison', 'distribution', 'breakdown', 
        'analyze', 'performance', 'statistics', 'show me'
    ]
    is_chart_request = any(k in user_query.lower() for k in chart_keywords)
    
    special_chart_instruction = ""
    if is_chart_request:
        special_chart_instruction = """
        6. **VISUALIZATION MODE**:
           - **YOU ARE PAIRED WITH A VISUALIZATION ENGINE**.
           - The user CAN see the chart.
           - **NEVER** say "I cannot generate charts" or "I am a text model".
           - Instead, say: "I have generated the chart below based on the data..." or "As you can see in the graph..."
           - Verify the data in the context matches the user's request, then describe the trend.
        """
        
    # Prepare previous conversation context
    history_context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in history[-4:]])

    prompt = f"""[SYSTEM: {current_system_prompt}]
    [HISTORY]:
    {history_context}
    
    [TASK: ANSWER QUESTION]
    [CONTEXT]:
    {retrieved_context[:5000]}
    
    [USER QUERY]: {user_query}
    (Refined Search: {search_query})
    
    INSTRUCTIONS:
    1. **INTERACTIVE STRUCTURE**: 
       - Use **Bold Headers** (start line with `### `).
       - Use **Bullet Points** for lists.
       - **Bold** key terms and numbers.
    2. **STRICT CITATION RULE**: 
       - format: `[Source: filename.pdf (Page X)]`
       - **Keep citation on the SAME LINE** as the text. Do not break lines inside citations.
    3. **IDENTITY**:
       - You are an intelligent document assistant.
       - If asked who built you, state that you are a document intelligence tool.
    4. **DATA COMPLETENESS**:
       - When asked for a list (e.g., student names), **SCAN ALL PROVIDED CONTEXTS**.
       - Do NOT stop after the first few items.
       - If the context lists 10 names across different pages, list all 10.
       - NEVER say "I can list more if you provide data" if the data is already present in any of the chunks below.
    5. **CALCULATIONS**:
       - Perform all average/sum/percentage calculations explicitly. Show the steps.
    5. **Format Example**:
       
       ### Student Performance
       * **Rohan**: 95 Marks `[Source: Class_Data.pdf (Page 1)]`
       * **Average**: The calculated average is **88.5** (derived from 20 entries).
    
    {special_chart_instruction}
    """

    # Persistent Auto-Save (Step 1: Save User Message)
    history.append({'role': 'user', 'content': user_query})
    executor.submit(save_to_firebase, session_id, history)

    def generate():
        chart_future = None
        
        # 1. Start Chart Generation in Parallel (Faster!)
        if is_chart_request:
            # Signal frontend to show placeholder/checklist
            yield f"data: {json.dumps({'step': 'chart', 'msg': '📊 Generating chart...'})}\n\n"
            # Submit to background thread (use search_query for better data extraction)
            chart_future = executor.submit(generate_chart, context, search_query)

        try:
            yield f"data: {json.dumps({'step': 'retrieving', 'msg': '🔍 Searching docs...'})}\n\n"
            yield f"data: {json.dumps({'step': 'analyzing', 'msg': '🧠 Analyzing content...'})}\n\n" 
            yield f"data: {json.dumps({'step': 'generating', 'msg': '⚡ Streaming response...'})}\n\n"
            
            # Gemini Streaming
            model = genai.GenerativeModel(CURRENT_MODEL)
            response_stream = model.generate_content(prompt, stream=True)
            
            yield f"data: {json.dumps({'step': 'done', 'msg': '✅ Complete'})}\n\n"
            
            full_text = ""
            for chunk in response_stream:
                if chunk.text:
                    full_text += chunk.text
                    yield f"data: {json.dumps({'text': chunk.text})}\n\n"
            
            # Save Assistant response to local and Firebase
            history.append({'role': 'assistant', 'content': full_text})
            executor.submit(save_to_firebase, session_id, history)
            
            # 4. Retrieve Chart Result
            if chart_future:
                try:
                    img = chart_future.result(timeout=15) # Increased timeout slightly
                    if img:
                         yield f"data: {json.dumps({'image': img})}\n\n"
                    else:
                         yield f"data: {json.dumps({'chart_error': 'Failed to generate chart'})}\n\n"
                except Exception as e:
                    print(f"Chart timeout/error: {e}")
                    yield f"data: {json.dumps({'chart_error': str(e)})}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
             yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/sessions', methods=['GET'])
def get_sessions():
    try:
        url = f"{FIREBASE_URL}/chats.json?shallow=true"
        response = requests.get(url)
        data = response.json()
        if data:
            return jsonify({"sessions": list(data.keys())})
        return jsonify({"sessions": []})
    except Exception as e:
        print(f"Firebase Session Error: {e}")
        return jsonify({"sessions": []})

@app.route('/session/<session_id>', methods=['GET'])
def get_session_details(session_id):
    try:
        url = f"{FIREBASE_URL}/chats/{session_id}.json"
        response = requests.get(url)
        data = response.json()
        if data:
            history = json.loads(data)
            return jsonify({"history": history})
        return jsonify({"error": "Session not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    try:
        url = f"{FIREBASE_URL}/chats/{session_id}.json"
        requests.delete(url)
        # Also clean up flashcards for this session if any
        requests.delete(f"{FIREBASE_URL}/flashcards/{session_id}.json")
        return jsonify({"message": "Session deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_flashcard', methods=['POST'])
def save_flashcard():
    try:
        data = request.json
        card = data.get('card')
        session_id = data.get('session_id', 'default_session')
        if not card: return jsonify({"error": "No card data"}), 400
        
        # We store saved cards per session or globally as requested
        # User said "/flashcards : { json data of all saved cards }"
        # Let's use a global or session-specific list. Let's do session-specific for better UX.
        url = f"{FIREBASE_URL}/flashcards/{session_id}.json"
        
        # Get existing first
        existing_resp = requests.get(url)
        existing = existing_resp.json()
        if existing:
            cards = json.loads(existing)
        else:
            cards = []
            
        # Add if not duplicate
        if card not in cards:
            cards.append(card)
            requests.put(url, json=json.dumps(cards))
            
        return jsonify({"message": "Saved", "count": len(cards)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_saved_flashcards', methods=['GET'])
def get_saved_flashcards():
    session_id = request.args.get('session_id', 'default_session')
    try:
        url = f"{FIREBASE_URL}/flashcards/{session_id}.json"
        response = requests.get(url)
        data = response.json()
        if data:
            return jsonify({"flashcards": json.loads(data)})
        return jsonify({"flashcards": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_title_from_questions(questions):
    try:
        # Extract a snippet to generate title from
        sample = json.dumps(questions[:3])
        prompt = f"""
        Generate a short, catchy 3-5 word title for a quiz containing these questions:
        {sample}
        OUTPUT ONLY THE TITLE. NO QUOTES.
        """
        model = genai.GenerativeModel(CURRENT_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip().replace('"', '').replace('**', '')
    except:
        return "General Knowledge Quiz"

@app.route('/get_all_quizzes', methods=['GET'])
def get_all_quizzes():
    from concurrent.futures import ThreadPoolExecutor
    try:
        # Fetch all data to process titles
        url = f"{FIREBASE_URL}/quizzes.json"
        response = requests.get(url)
        data = response.json()
        
        if not data: return jsonify({"quizzes": []})
        
        quiz_list = []
        updates = {}
        
        for qid, qcontent in data.items():
            title = "Untitled Quiz"
            
            # Handle migration from List -> Dict
            if isinstance(qcontent, list):
                title = generate_title_from_questions(qcontent)
                new_struct = {"title": title, "questions": qcontent}
                updates[qid] = new_struct
            elif isinstance(qcontent, dict):
                title = qcontent.get("title", "Untitled Quiz")
                # If title is missing or default
                if title == "Untitled Quiz" and "questions" in qcontent:
                    title = generate_title_from_questions(qcontent["questions"])
                    qcontent["title"] = title
                    updates[qid] = qcontent
            
            quiz_list.append({"id": qid, "title": title})
            
        # Batch update Firebase (using thread pool for speed if many)
        if updates:
            def update_fb(item):
                k, v = item
                requests.put(f"{FIREBASE_URL}/quizzes/{k}.json", json=v)
            
            with ThreadPoolExecutor(max_workers=5) as pool:
                pool.map(update_fb, updates.items())
                
        return jsonify({"quizzes": quiz_list})
    except Exception as e:
        print(f"Error fetching quizzes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    if vector_index is None: return jsonify({"error": "No documents uploaded."}), 400
    
    # Get a broad sample of context
    context_chunks = retrieve_chunks("General summary and key facts", k=20)
    context = "\n".join([c['text'] for c in context_chunks])
    
    prompt = f"""
    [TASK: GENERATE QUIZ]
    Based on the provided context, generate a 10-question multiple choice quiz.
    
    CONTEXT:
    {context[:12000]}
    
    OUTPUT FORMAT (JSON):
    {{
      "title": "A Creative Title for this Quiz",
      "questions": [
          {{
            "question": "...",
            "options": ["A", "B", "C", "D"],
            "answer": "Correct Option"
          }},
          ...
      ]
    }}
    """
    try:
        model = genai.GenerativeModel(CURRENT_MODEL)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        quiz_data = json.loads(response.text)
        
        # Save to Firebase
        quiz_id = str(uuid.uuid4())
        try:
            url = f"{FIREBASE_URL}/quizzes/{quiz_id}.json"
            requests.put(url, json=quiz_data)
        except Exception as e:
            print(f"Firebase Save Error: {e}")
            
        return jsonify({"quiz_id": quiz_id, "quiz": quiz_data}) # quiz_data now includes title
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_quiz/<quiz_id>', methods=['GET'])
def get_quiz(quiz_id):
    try:
        url = f"{FIREBASE_URL}/quizzes/{quiz_id}.json"
        response = requests.get(url)
        data = response.json()
        if data:
            return jsonify({"quiz": data})
        return jsonify({"error": "Quiz not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/generate_flashcards', methods=['POST'])
def generate_flashcards():
    if vector_index is None: return jsonify({"error": "No documents uploaded."}), 400
    
    context_chunks = retrieve_chunks("Key terms and definitions", k=15)
    context = "\n".join([c['text'] for c in context_chunks])
    
    prompt = f"""
    [TASK: GENERATE FLASHCARDS]
    Extract 8 key terms and their concise definitions/details from the context.
    
    CONTEXT:
    {context[:10000]}
    
    OUTPUT FORMAT (JSON):
    [
      {{
        "term": "...",
        "definition": "..."
      }},
      ...
    ]
    """
    try:
        model = genai.GenerativeModel(CURRENT_MODEL)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        # Robust JSON cleaning
        text = response.text.strip()
        if text.startswith("```json"): text = text[7:]
        if text.endswith("```"): text = text[:-3]
        
        try:
            flashcards = json.loads(text)
        except:
            # Fallback repair
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match: 
                flashcards = json.loads(match.group(0))
            else:
                raise ValueError("Could not phrase JSON from response")

        return jsonify(flashcards)
    except Exception as e:
        print(f"❌ Flashcard Generation Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/deep_analysis', methods=['POST'])
def deep_analysis():
    if vector_index is None: return jsonify({"error": "No documents uploaded."}), 400
    
    context_chunks = retrieve_chunks("Comprehensive overview and critical insights", k=20)
    context = "\n".join([c['text'] for c in context_chunks])
    
    prompt = f"""
    [TASK: DEEP DOCUMENT ANALYSIS]
    Provide a comprehensive analysis of the document content including:
    1. Executive Summary
    2. Key Findings & Quantitative Data
    3. Critical Analysis/Implications
    4. Strategic Recommendations or Next Steps
    
    CONTEXT:
    {context[:15000]}
    
    OUTPUT:
    Markdown format with ### Headers.
    """
    try:
        model = genai.GenerativeModel(CURRENT_MODEL)
        response = model.generate_content(prompt)
        return jsonify({"analysis": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Only run initialization in the main process (not the reloader)
    import os
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true' or not app.debug:
        print("="*50)
        print("🚀 Initializing RAG Application...")
        print("="*50)
        sys.stdout.flush()
        
        load_vector_db()
        
        print("="*50)
        print("✅ Application Ready!")
        print("="*50)
        sys.stdout.flush()
    
    app.run(debug=False, host='0.0.0.0', port=7860)

