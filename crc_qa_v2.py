import os
import pickle
import requests
import sys
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import re
import json
import time

# --- Global Constants and Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE_PATH = os.path.join(BASE_DIR, 'code', 'index.pkl')
BIB_DB_PATH = os.path.join(BASE_DIR, 'bib_db.json')

if not os.path.exists(INDEX_FILE_PATH):
    INDEX_FILE_PATH = os.path.join(BASE_DIR, 'index.pkl')

print(f"Resolved INDEX_FILE_PATH: {INDEX_FILE_PATH}")

# Load Bib DB
BIB_DB = {}
if os.path.exists(BIB_DB_PATH):
    try:
        with open(BIB_DB_PATH, 'r', encoding='utf-8') as f:
            BIB_DB = json.load(f)
        print(f"Loaded BIB_DB with {len(BIB_DB)} entries.")
    except Exception as e:
        print(f"Failed to load BIB_DB: {e}")

CITATION_STYLE = "apa" # or "gbt"

EMBEDDING_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
TOP_K_FOR_CONTEXT = 8
TOP_N_FINAL = 8

def refine_evidence(chunks, top_n=8):
    """
    Refine the gathered evidence chunks:
    1. Filter out chunks with missing metadata or bad sources.
    2. Rerank/sort (currently by existing score if available, or just keeping the retrieval order).
    3. Truncate to top_n.
    """
    valid_chunks = []
    for c in chunks:
        src = c.get('source', '')
        # Strict Reference Check: Must have source
        if not src:
            continue
            
        # Extra source integrity rules
        if '...' in src: # Ellipsis indicating truncation
            continue
        if len(src) < 5: # Too short
            continue
        if "not fully cited" in src.lower():
            continue
            
        valid_chunks.append(c)
            
    # Rerank logic: 
    # If chunks have 'score', sort by it. 
    valid_chunks.sort(key=lambda x: x.get('score', 0), reverse=True)
    
    return valid_chunks[:top_n]

def search_literature_two_stage(query, index_file_path, top_k=10):
    if not os.path.exists(index_file_path):
        return []

    with open(index_file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        all_chunks = data.get('chunks', [])
        all_embeddings = data.get('embeddings', [])
    elif isinstance(data, list):
        return []
    else:
        return []
        
    if all_embeddings is None or len(all_embeddings) == 0:
        return []

    # Identify categories
    biomarker_keywords = ["biomarker", "biomarkers", "marker", "tumor marker", "molecular marker"]
    is_biomarker_query = any(kw in query.lower() for kw in biomarker_keywords)
    
    diagnosis_keywords = ["early diagnosis", "early detection", "screening", "non-invasive", "stool test", "stool dna", 
                          "mir-21", "mir-92a", "microrna", "mirna"]
    is_diagnosis_query = any(kw in query.lower() for kw in diagnosis_keywords)

    treatment_keywords = [
        "treatment", "therapy", "systemic therapy", "chemotherapy", "regimen", "first-line", "second-line", 
        "folfox", "folfiri", "capox", "xelox", "bevacizumab", "cetuximab", "panitumumab", 
        "immunotherapy", "checkpoint inhibitor", "pd-1", "nivolumab", "pembrolizumab"
    ]
    is_treatment_query = any(kw in query.lower() for kw in treatment_keywords)

    query_embedding = EMBEDDING_MODEL.encode(query, convert_to_tensor=True)
    
    if not isinstance(all_embeddings, torch.Tensor):
        if isinstance(all_embeddings, list):
            all_embeddings = np.array(all_embeddings)
        all_embeddings = torch.tensor(all_embeddings)
    
    all_embeddings = all_embeddings.to(query_embedding.device)

    top_results_indices = []
    top_results_scores = []

    # Priority Strategy
    target_category = None
    if is_treatment_query: target_category = 'treatment'
    elif is_diagnosis_query: target_category = 'diagnosis'
    elif is_biomarker_query: target_category = 'biomarker'

    if target_category:
        cat_indices = [i for i, chunk in enumerate(all_chunks) if chunk.get('category') == target_category]
        if cat_indices:
            cat_embeddings = all_embeddings[cat_indices]
            cosine_scores = util.pytorch_cos_sim(query_embedding, cat_embeddings)[0]
            k_candidates = min(30, len(cat_indices))
            top_k_results = cosine_scores.topk(k=k_candidates)
            candidate_indices = [cat_indices[idx.item()] for idx in top_k_results[1]]
            candidate_scores = top_k_results[0]
            
            if len(candidate_indices) >= top_k:
                top_results_indices = candidate_indices
                top_results_scores = candidate_scores

    # Fallback to full search
    if not top_results_indices:
        cosine_scores = util.pytorch_cos_sim(query_embedding, all_embeddings)[0]
        top_k_results = cosine_scores.topk(k=top_k * 3)
        top_results_indices = top_k_results[1].tolist()
        top_results_scores = top_k_results[0]

    retrieved_docs = []
    for score, idx in zip(top_results_scores, top_results_indices):
        idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
        score_val = score.item() if isinstance(score, torch.Tensor) else score

        doc_info = {
            'text': all_chunks[idx_val]['text'],
            'source': all_chunks[idx_val]['source'],
            'score': score_val,
            'file': all_chunks[idx_val].get('file', all_chunks[idx_val]['source']),
            'doc_id': all_chunks[idx_val].get('doc_id', all_chunks[idx_val].get('file', all_chunks[idx_val]['source'])),
            'page': all_chunks[idx_val].get('page', '?'),
            'chunk_id': all_chunks[idx_val].get('chunk_id', '?')
        }
        if 'category' in all_chunks[idx_val]:
            doc_info['category'] = all_chunks[idx_val]['category']
        if 'subtype' in all_chunks[idx_val]:
            doc_info['subtype'] = all_chunks[idx_val]['subtype']
            
        retrieved_docs.append(doc_info)
        
    # Keyword Reranking
    query_keywords = set(query.lower().split())
    
    biomarker_specific = {"cea", "ca19-9", "msi", "dmmr", "kras", "nras", "braf", "her2", "ctdna"}
    if is_biomarker_query: query_keywords.update(biomarker_specific)

    diagnosis_specific = {"mir-21", "mir-92a", "mirna", "microrna", "stool", "dna", "fit", "screening", "early"}
    if is_diagnosis_query: query_keywords.update(diagnosis_specific)

    treatment_specific = {"folfox", "folfiri", "capox", "xelox", "bevacizumab", "cetuximab", "panitumumab", "pembrolizumab", "nivolumab", "first-line", "second-line", "chemotherapy", "immunotherapy"}
    if is_treatment_query: query_keywords.update(treatment_specific)
    
    def rank_by_keyword_overlap(doc):
        doc_keywords = set(doc['text'].lower().split())
        overlap = len(query_keywords.intersection(doc_keywords))
        
        if is_treatment_query and doc.get('category') == 'treatment': overlap += 5
        elif is_diagnosis_query and doc.get('category') == 'diagnosis':
            overlap += 5
            if doc.get('subtype') == 'miRNA_early_diagnosis' and any(k in query.lower() for k in ["mir", "rna"]):
                 overlap += 3
        elif is_biomarker_query and doc.get('category') == 'biomarker': overlap += 5
            
        doc_text_lower = doc['text'].lower()
        if "biomarker" in doc_text_lower: overlap += 1
        if "diagnosis" in doc_text_lower or "screening" in doc_text_lower: overlap += 1
        if "treatment" in doc_text_lower or "therapy" in doc_text_lower: overlap += 1
            
        return overlap

    reranked_docs = sorted(retrieved_docs, key=rank_by_keyword_overlap, reverse=True)
    return reranked_docs[:top_k]

def search_literature_keyword(query, index_file_path, top_k=5):
    """
    Pure keyword-based search to complement vector search.
    Scans all chunks and ranks by keyword overlap with the query.
    """
    if not os.path.exists(index_file_path):
        return []

    with open(index_file_path, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        all_chunks = data.get('chunks', [])
    else:
        return []

    query_keywords = set(query.lower().split())
    if not query_keywords:
        return []

    # Filter and score
    candidates = []
    for chunk in all_chunks:
        text_lower = chunk['text'].lower()
        chunk_keywords = set(text_lower.split())
        overlap = len(query_keywords.intersection(chunk_keywords))
        
        # Basic scoring: overlap count
        if overlap > 0:
            candidates.append({
                'chunk': chunk,
                'score': overlap
            })
    
    # Sort by score descending
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # Format results
    results = []
    for item in candidates[:top_k]:
        c = item['chunk']
        results.append({
            'text': c['text'],
            'source': c['source'],
            'score': item['score'],
            'category': c.get('category'),
            'subtype': c.get('subtype'),
            'file': c.get('file', c['source']),
            'doc_id': c.get('doc_id', c.get('file', c['source'])),
            'page': c.get('page', '?'),
            'chunk_id': c.get('chunk_id', '?')
        })
        
    return results

from requests.exceptions import Timeout, RequestException

def call_deepseek_api(messages, api_key=None, stream=False):
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("[API] Error: DEEPSEEK_API_KEY not found in env or args")
        return "Error: DEEPSEEK_API_KEY not set"

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": stream,
        "max_tokens": 2048,
        "temperature": 0.3
    }

    try:
        # Timeout: 5s connect, 30s read
        response = requests.post(url, headers=headers, json=payload, timeout=(5, 30))
        response.raise_for_status()
        if stream:
            return response
        return response.json()['choices'][0]['message']['content']
    except Timeout:
        print("[API] TimeoutError: DeepSeek API connection/read timed out.")
        return "Error: DeepSeek API timed out (Connect 5s, Read 30s)."
    except RequestException as e:
        print(f"[API] RequestException: {e}")
        return f"Error: DeepSeek API failed - {str(e)}"
    except Exception as e:
        print(f"API Error: {e}")
        return str(e)

def call_deepseek_api_stream(messages, api_key=None):
    if not api_key:
        api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        print("[API] Error: DEEPSEEK_API_KEY not found in env or args")
        yield "Error: DEEPSEEK_API_KEY not set"
        return

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": True,
        "max_tokens": 2048,
        "temperature": 0.3
    }
    
    try:
        # Timeout: 5s connect, 30s read
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=(5, 30))
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    json_str = line_str[6:]
                    if json_str == '[DONE]': break
                    try:
                        data = json.loads(json_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                yield delta['content']
                    except: continue
    except Timeout:
        yield " [Error: DeepSeek API request timed out]"
    except RequestException as e:
        yield f" [Error: Network failure: {str(e)}]"
    except Exception as e:
        yield f"Error: {str(e)}"

def generate_subqueries(query, api_key):
    prompt = f"""
    Analyze the following medical question and break it down into 2-4 specific search subqueries.
    Also identify the key required points that must be covered in the answer.
    Output ONLY a JSON object with the following format:
    {{
        "subqueries": ["query1", "query2"],
        "required_points": ["point1", "point2"]
    }}
    
    Question: {query}
    """
    messages = [{"role": "user", "content": prompt}]
    response = call_deepseek_api(messages, api_key=api_key)
    try:
        # Extract JSON from response if wrapped in code blocks
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(response)
    except:
        return {"subqueries": [query], "required_points": []}

def append_refs(answer_text, retrieved_docs):
    """
    Append formatted references to the answer text based on retrieved docs.
    """
    used_doc_ids = []
    
    # Extract citations from the answer text to filter used docs
    # Look for patterns like (Author et al., Year
    cited_patterns = re.findall(r'\(([^,]+)(?: et al\.)?,?\s*(\d{4})', answer_text)
    
    # Create a mapping of "Author-Year" -> [doc_ids]
    # Because multiple docs might have same Author-Year (e.g. "Smith et al., 2021")
    # we need to be careful. But generally, citation_short is unique enough or we just match roughly.
    
    # Candidate mapping from retrieved docs
    cand = {} # key -> [doc_id, ...]
    for d in retrieved_docs:
        doc_id = d.get('doc_id') or d.get('file') or d.get('source')
        # Get the short citation key from BIB_DB
        entry = BIB_DB.get(doc_id, {})
        k = (entry.get("citation_short") or "").lower()
        if k:
            cand.setdefault(k, []).append(doc_id)
            
    # If no citations found (fallback), show all retrieved.
    if not cited_patterns:
        # Fallback to showing all retrieved (original logic)
        for d in retrieved_docs:
            doc_id = d.get('doc_id') or d.get('file') or d.get('source')
            if doc_id and doc_id not in used_doc_ids:
                used_doc_ids.append(doc_id)
    else:
        # Filter retrieved docs that match the citations
        # We construct keys from the extracted (Author, Year) and look them up in 'cand'
        for (auth, yr) in cited_patterns:
            # Try both "et al." and single author formats
            # The regex capture group 1 includes everything before the comma (e.g. "Smith" or "Smith et al" is handled by regex?)
            # Wait, regex was: r'\(([A-Za-z\u4e00-\u9fff\-']+)(?:\s+et al\.)?,\s*((?:19|20)\d{2}|n\.d\.)\)'
            # Group 1 is just the name (e.g. "Smith").
            
            # Construct possible keys that match BIB_DB's citation_short format
            # BIB_DB citation_short is usually "Smith et al., 2021" or "Smith, 2021"
            
            k1 = f"{auth} et al., {yr}".lower()
            k2 = f"{auth}, {yr}".lower()
            
            found_docs = cand.get(k1, []) + cand.get(k2, [])
            
            for doc_id in found_docs:
                if doc_id not in used_doc_ids:
                    used_doc_ids.append(doc_id)
        
        # If filtering resulted in 0 docs (e.g. LLM hallucinated citations or format mismatch), 
        # fallback to all retrieved to be safe.
        if not used_doc_ids:
             for d in retrieved_docs:
                doc_id = d.get('doc_id') or d.get('file') or d.get('source')
                if doc_id and doc_id not in used_doc_ids:
                    used_doc_ids.append(doc_id)

    if not used_doc_ids:
        return answer_text
        
    lines = []
    for doc_id in used_doc_ids:
        e = BIB_DB.get(doc_id, {"file": doc_id})
        line = e.get("citation_gbt" if CITATION_STYLE == "gbt" else "citation_apa")
        if not line:
            au = ", ".join(e.get("authors") or ["Unknown"])
            y = e.get("year") or "n.d."
            t = e.get("title") or e.get("file") or doc_id
            line = f"{au}. ({y}). {t}."
        lines.append(line)
        
    return answer_text + "\n\nReferences\n- " + "\n- ".join(lines)

def answer_crc_question(user_question: str, api_key: str = None, mode: str = "vanilla"):
    start_time = time.time()
    print(f"[RAG] start pid={os.getpid()} python={sys.executable} time={start_time}")
    print(f"[RAG] query length={len(user_question)} (first 120: {user_question[:120]})")
    print(f"[RAG] config: db_path={INDEX_FILE_PATH}, exists={os.path.exists(INDEX_FILE_PATH)}, api_key_present={bool(api_key)}")
    print(f"[RAG] retrieval start")
    
    log_entry = {
        "question": user_question,
        "mode": mode,
        "retrieval_rounds": 0,
        "subqueries": [],
        "retrieved_doc_ids": [],
        "candidate_chunks_total": 0,
        "final_used_chunks": 0,
        "num_chunks": 0,
        "fallback": False,
        "time_taken": 0
    }
    
    unique_chunks = []
    seen_texts = set()

    def process_chunks(chunks):
        new_count = 0
        for chunk in chunks:
            cleaned_text = ' '.join(chunk['text'].split())
            if cleaned_text not in seen_texts:
                seen_texts.add(cleaned_text)
                chunk['text'] = cleaned_text
                unique_chunks.append(chunk)
                log_entry["retrieved_doc_ids"].append(chunk['source'])
                new_count += 1
        return new_count

    # Round 1 Retrieval
    log_entry["retrieval_rounds"] += 1
    chunks = search_literature_two_stage(user_question, INDEX_FILE_PATH, top_k=TOP_K_FOR_CONTEXT)
    process_chunks(chunks)
    print(f"[RAG] retrieval done in {time.time()-start_time:.4f}s, got {len(chunks)} chunks (Round 1)")

    # Agentic Mode Logic
    if mode == "agentic":
        # CIN Expansion Logic
        cin_keywords = ["chromosomal instability", "aneuploidy", "somatic copy-number alteration", "scna", "copy-number alteration", "mss cin"]
        is_cin_query = any(k in user_question.lower() for k in ["cin", "chromosomal instability", "aneuploidy"])
        
        plan = generate_subqueries(user_question, api_key)
        subqueries = plan.get("subqueries", [])
        
        if is_cin_query:
            print("CIN related query detected. Injecting synonym keywords...")
            # Append a special subquery for keyword search
            subqueries.append(" ".join(cin_keywords))
            
        log_entry["subqueries"] = subqueries
        
        # Force 2-round execution as requested
        # Also supports "insufficient evidence" triggers (heuristic or answer self-check)
        # Since we are forcing, we skip the intermediate answer generation to save time.
        
        trigger_round_2 = True # Forced
        
        if trigger_round_2:
            print("Agentic Mode: Triggering Round 2 (Decomposition + Multi-strategy Retrieval)...")
            log_entry["retrieval_rounds"] += 1
            
            for subq in log_entry["subqueries"]:
                # Strategy 1: Vector Search (standard)
                vec_chunks = search_literature_two_stage(subq, INDEX_FILE_PATH, top_k=3)
                process_chunks(vec_chunks)
                
                # Strategy 2: Keyword Search (different retriever)
                kw_chunks = search_literature_keyword(subq, INDEX_FILE_PATH, top_k=3)
                process_chunks(kw_chunks)
                
    # Record total candidates before refinement
    log_entry["candidate_chunks_total"] = len(unique_chunks)
    
    # Refine Evidence (for both modes, or just agentic? User said "A) ... agentic ... add refine_evidence")
    # But doing it for both ensures consistency and reference safety.
    # However, strict instructions were under Agentic. Let's apply globally for safety but definitely for agentic.
    # Actually, vanilla mode usually returns TOP_K (8), so refinement (8) doesn't change much unless some are invalid.
    # Let's apply to all.
    
    unique_chunks = refine_evidence(unique_chunks, top_n=TOP_N_FINAL)
    log_entry["final_used_chunks"] = len(unique_chunks)
    log_entry["final_unique_sources"] = len(set(c['source'] for c in unique_chunks))
    log_entry["num_chunks"] = len(unique_chunks)
    
    print(f"Total candidates: {log_entry['candidate_chunks_total']}, Final used: {log_entry['final_used_chunks']}")
    
    if len(unique_chunks) == 0:
        # Fallback to general answer mode if no evidence
        log_entry["fallback"] = True
        log_entry["missing_evidence"] = True
        
        effective_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not effective_api_key:
             print("[RAG] No evidence and no API key.")
             yield "No relevant internal documents found."
             log_entry["time_taken"] = time.time() - start_time
             yield log_entry
             return

        system_prompt = "You are a helpful assistant. Provide a general medical overview clearly and directly. Do NOT include citations."
        user_prompt = f"Please provide a general overview of this topic without citing specific sources: {user_question}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print(f"[RAG] generation start (fallback)")
        t_gen = time.time()
        
        accumulated_answer = ""
        try:
            for chunk in call_deepseek_api_stream(messages, api_key=api_key):
                accumulated_answer += chunk
                yield accumulated_answer
            print(f"[RAG] generation done in {time.time()-t_gen:.4f}s, output length={len(accumulated_answer)}")
        except Exception as e:
            print(f"[RAG] ERROR in fallback generation: {e}")
            import traceback
            traceback.print_exc()
            yield f"[Error] {e}"
            
        log_entry["time_taken"] = time.time() - start_time
        yield log_entry
        return

    # Construct Prompt
    context_str = ""
    # We will use explicit file/page citations instead of numbered list mapping
    for chunk in unique_chunks:
        file_name = chunk.get('file', chunk['source'])
        doc_id = chunk.get('doc_id') or file_name
        page = chunk.get('page', '?')
        cid = chunk.get('chunk_id', '?')
        
        # Look up citation info
        entry = BIB_DB.get(doc_id, {})
        short_cit = entry.get("citation_short") or file_name
        
        # Header format: Source: Author et al., Year (doc_id:..., chunk_id:...)
        # Removed page number from header to avoid confusing the model into citing p.X
        ref_header = f"{short_cit} (doc_id:{doc_id}, chunk_id:{cid})"
        
        context_str += f"Source: {ref_header}\n{chunk['text']}\n\n"

    system_prompt = (
        "Answer the question directly. Do NOT add any background explanation like 'based on the provided context' or 'from the literature'. Simply provide the conclusion with any necessary explanation.\n"
        "If the context is relevant, use it to answer the question to the best of your ability. "
        "Do not state 'insufficient evidence' or 'no evidence retrieved' unless the context contains absolutely no relevant information.\n"
        "Citation Rules:\n"
        "1. Every claim must be cited using the format (Author et al., Year) at the end of the sentence.\n"
        "2. The Author/Year must be derived from the Source header of the context block.\n"
        "3. Do NOT use [1], [2] numbers.\n"
        "4. Every paragraph must contain at least one valid citation in (Author et al., Year) format.\n"
        "Structure: Introduction, Details, Conclusion.\n"
        "Do NOT output a References section; references will be appended programmatically.\n"
        "IMPORTANT: You must ONLY use the source titles/pages provided in the context blocks. Do not hallucinate."
    )
    
    # Check for API key availability
    effective_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    
    if not effective_api_key:
        print("[RAG] No API key found. Returning context for external generation.")
        yield f"**Retrieved Context:**\n\n{context_str}"
        log_entry["time_taken"] = time.time() - start_time
        yield log_entry
        return

    user_prompt = f"Question: {user_question}\n\nContext:\n{context_str}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    print(f"[RAG] generation start (normal)")
    t_gen = time.time()

    accumulated_answer = ""
    # Ensure api_key is passed to call_deepseek_api_stream
    try:
        for chunk in call_deepseek_api_stream(messages, api_key=api_key):
            accumulated_answer += chunk
            yield accumulated_answer
        print(f"[RAG] generation done in {time.time()-t_gen:.4f}s, output length={len(accumulated_answer)}")
    except Exception as e:
        print(f"[RAG] ERROR in normal generation: {e}")
        import traceback
        traceback.print_exc()
        yield f"[Error] {e}"
        
    # Clean up any residual p.X citations from the model output just in case
    import re
    accumulated_answer = re.sub(r"(\([^\(\)]*?),\s*p\.\s*\d+(\))", r"\1\2", accumulated_answer)
    
    # After generation complete, append full references
    full_answer_with_refs = append_refs(accumulated_answer, unique_chunks)
    yield full_answer_with_refs
        
    log_entry["time_taken"] = time.time() - start_time
    yield log_entry

def answer_crc_question_sync(user_question: str, api_key: str = None, mode: str = "vanilla"):
    full_answer = ""
    log_data = {}
    
    gen = answer_crc_question(user_question, api_key=api_key, mode=mode)
    for item in gen:
        if isinstance(item, dict):
            log_data = item
        else:
            full_answer = item
            
    return full_answer, log_data

TEST_QUESTIONS = [
    "What are the main strategies for early detection and screening of colorectal cancer, and how do they differ in terms of sensitivity, specificity, and invasiveness?",
    "How do fecal immunochemical tests (FIT) compare with multitarget stool DNA testing (mt-sDNA) for colorectal cancer screening in terms of performance and clinical use?",
    "How are circulating miRNA biomarkers used for the early detection of colorectal cancer and advanced adenomas?",
    "How do liquid biopsy techniques compare with traditional biopsy in the diagnosis and monitoring of colorectal cancer?",
    "What are the standard first-line systemic treatment options for metastatic colorectal cancer, and how are regimens selected for different patient subgroups?",
    "What are the typical second-line treatment options after progression on first-line therapy in metastatic colorectal cancer?",
    "How is MSI-H/dMMR status used to guide treatment decisions in metastatic colorectal cancer, particularly with respect to immunotherapy?",
    "How do treatment strategies differ between RAS wild-type and RAS-mutant metastatic colorectal cancer, especially regarding anti-EGFR and anti-VEGF therapies?",
    "What are the potential benefits of targeted therapies in treating metastatic colorectal cancer, and which are currently approved?",
    "How do chemotherapy and immunotherapy compare in treating metastatic colorectal cancer?",
    "What are the latest advancements in personalized treatment approaches for colorectal cancer?",
    "What are the most important tumor and molecular biomarkers in colorectal cancer, and how are they used for diagnosis, prognosis, and treatment selection?",
    "How are serum CEA and CA19-9 used in the management of colorectal cancer, and what are their main limitations?",
    "What roles do KRAS/NRAS, BRAF V600E, and HER2 amplification play as predictive biomarkers in metastatic colorectal cancer?",
    "How is circulating tumor DNA (ctDNA) used in colorectal cancer for minimal residual disease (MRD) assessment and recurrence monitoring?",
    "How are genetic mutations in colorectal cancer used to predict treatment response and patient outcomes?",
    "What is the current global burden of colorectal cancer, and how do incidence and mortality vary by region, age, and sex?",
    "What are the major modifiable and non-modifiable risk factors for colorectal cancer?",
    "What characterizes early-onset colorectal cancer (diagnosed before age 50), and what are the proposed risk factors and clinical challenges?",
    "How does a family history of colorectal cancer impact individual risk, and what preventive measures can be taken?",
    "What is the role of diet and lifestyle in colorectal cancer risk, and how can these factors be modified for prevention?",
    "How does obesity influence colorectal cancer risk, and what are the mechanisms behind this association?",
    "How effective are lifestyle changes in preventing colorectal cancer, and what interventions are recommended?",
    "What role does genetic counseling play in the prevention of colorectal cancer in individuals with a family history?",
    "What are the current guideline recommendations for average-risk colorectal cancer screening in terms of starting age, stopping age, and test intervals?"
]
