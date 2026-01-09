import time
import math
from typing import List, Dict, Set, Optional
from botocore.exceptions import ClientError
from app.core.config import settings
from app.core.aws import s3_vectors_client

def _put_vectors_batch(batch: List[Dict], attempt: int = 1):
    try:
        s3_vectors_client.put_vectors(
            vectorBucketName=settings.VECTOR_BUCKET_NAME,
            indexName=settings.INDEX_NAME,
            vectors=batch,
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in {"Throttling", "ThrottlingException"} and attempt <= settings.VECTOR_STORAGE_MAX_RETRIES:
            backoff = (settings.VECTOR_STORAGE_BACKOFF_FACTOR ** (attempt - 1))
            print(f"Throttled. Retrying attempt {attempt + 1} in {backoff} seconds...")
            time.sleep(backoff)
            _put_vectors_batch(batch, attempt + 1)
        else:
            print(f"Failed to put vectors into S3: {e}")
            raise

def store_vectors_in_s3(processed_chunks: List[Dict], embeddings: List[List[float]], filename: str):
    if not processed_chunks or len(processed_chunks) != len(embeddings):
        raise ValueError("Mismatch between chunks and embeddings count.")

    s3_key_base = filename.rsplit('.', 1)[0]
    vectors_to_store = []

    for i, chunk_data in enumerate(processed_chunks):
        chunk_text = chunk_data.get('content', '')
        if not chunk_text:
            continue # Skip empty chunks
            
        vector_key = f"{s3_key_base}-chunk{i:04d}"
 
        chunk_type = chunk_data.get('type', 'general_content')
        
        metadata = {
            "filename": filename[:60],
            "doc_id": s3_key_base[:120],
            "chunk_index": str(i),
            "source_text": chunk_text,
            "chunk_content_type": chunk_type,
        }

        if "page_number" in chunk_data and chunk_data.get("page_number") is not None:
            metadata["page_number"] = str(chunk_data.get("page_number"))
        if "section_path" in chunk_data and chunk_data.get("section_path"):
            metadata["section_path"] = str(chunk_data.get("section_path"))[:300]
        
        vectors_to_store.append({
            "key": vector_key,
            "data": {"float32": embeddings[i]},
            "metadata": metadata,
        })

    for i in range(0, len(vectors_to_store), settings.VECTOR_PUT_BATCH_SIZE):
        batch = vectors_to_store[i:i + settings.VECTOR_PUT_BATCH_SIZE]
        _put_vectors_batch(batch)

    print(f"Successfully stored {len(vectors_to_store)} vectors in index: {settings.INDEX_NAME}")
    return len(vectors_to_store)


def query_vectors(query_embedding: List[float], top_k: int = 3) -> List[Dict]:
    try:
        top_k = max(1, min(int(settings.RETRIEVAL_MAX_CANDIDATES), int(top_k)))
        response = s3_vectors_client.query_vectors(
            vectorBucketName=settings.VECTOR_BUCKET_NAME,
            indexName=settings.INDEX_NAME,
            queryVector={"float32": query_embedding},
            topK=top_k,
            returnMetadata=True
        )
        return response.get("vectors", [])
    except Exception as e:
        print(f"Error querying vectors from S3: {e}")
        raise

def rerank_results(results: List[Dict], query_text: str) -> List[Dict]:
    query_lower = (query_text or "").lower()

    is_table_query = any(keyword in query_lower for keyword in [
        'table', 'data', 'figure', 'chart', 'row', 'column', 'value'
    ])
    is_definition_query = any(keyword in query_lower for keyword in [
        'what is', 'define', 'definition', 'meaning', 'explain'
    ])
    is_procedure_query = any(keyword in query_lower for keyword in [
        'how to', 'steps', 'process', 'procedure', 'method'
    ])

    for result in results:
        metadata = result.get("metadata", {})
        content = (metadata.get("source_text") or "").lower()
        content_type = metadata.get("chunk_content_type", "general_content")

        base_score = _base_relevance(result)

        type_boost = 0.0
        if is_table_query and content_type == "table_row":
            type_boost = 0.4
        elif is_definition_query and content_type == "definition":
            type_boost = 0.3
        elif is_procedure_query and content_type == "procedure":
            type_boost = 0.3

        table_boost = 0.0
        if '[table start]' in content or 'headers:' in content or 'row:' in content:
            table_boost = 0.2 if is_table_query else 0.1

        keyword_boost = 0.0
        query_keywords = _tokenize(query_lower)
        content_keywords = _tokenize(content)
        common_keywords = query_keywords.intersection(content_keywords)
        if common_keywords:
            keyword_boost = min(0.2, len(common_keywords) * 0.05)

        final_score = base_score * (1 + type_boost + table_boost + keyword_boost)
        result["rerank_score"] = final_score

    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return results


def query_vectors_hybrid(
    query_text: str,
    query_embedding: List[float],
    top_k: int = 5,
    candidate_k: int = 40,
    alpha: float = 0.7,
    lambda_mmr: float = 0.5,
    doc_ids: Optional[List[str]] = None,
) -> List[Dict]:

    candidate_k = max(5, min(int(settings.RETRIEVAL_MAX_CANDIDATES), int(candidate_k)))
    candidates = query_vectors(query_embedding, top_k=candidate_k)
    if not candidates:
        return []

    if doc_ids:
        allowed = [d for d in doc_ids if d]
        if allowed:
            prefixes = tuple(f"{d}-chunk" for d in allowed)
            filtered: List[Dict] = []
            for c in candidates:
                key = str(c.get("key") or "")
                meta = c.get("metadata", {}) or {}
                doc_id = str(meta.get("doc_id") or "")
                if (doc_id and doc_id in allowed) or (key and key.startswith(prefixes)):
                    filtered.append(c)
            candidates = filtered
    if not candidates:
        return []

    base_sims = [_base_relevance(c) for c in candidates]
    min_s, max_s = min(base_sims), max(base_sims)
    def norm_sim(s: float) -> float:
        if max_s == min_s:
            return 1.0
        return (s - min_s) / (max_s - min_s)

    q_tokens = list(_tokenize_list(query_text))
    doc_tokens_list: List[List[str]] = []
    for c in candidates:
        meta = c.get("metadata", {})
        content = meta.get("source_text", meta.get("source_text_preview", "")) or ""
        doc_tokens_list.append(list(_tokenize_list(content)))

    def bm25_scores() -> List[float]:
        if not q_tokens:
            return [0.0 for _ in candidates]
        N = len(doc_tokens_list)
        if N == 0:
            return []
        df: Dict[str, int] = {}
        for toks in doc_tokens_list:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        idf: Dict[str, float] = {}
        for t, f in df.items():
            idf[t] = math.log(1.0 + (N - f + 0.5) / (f + 0.5))

        avgdl = sum(len(toks) for toks in doc_tokens_list) / max(1, N)
        k1 = 1.2
        b = 0.75
        out: List[float] = []
        for toks in doc_tokens_list:
            if not toks:
                out.append(0.0)
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = len(toks)
            denom_norm = k1 * (1.0 - b + b * (dl / max(1e-6, avgdl)))
            score = 0.0
            for qt in q_tokens:
                f = tf.get(qt, 0)
                if f <= 0:
                    continue
                score += idf.get(qt, 0.0) * (f * (k1 + 1.0)) / (f + denom_norm)
            out.append(score)
        return out

    lex_raw = bm25_scores()
    if lex_raw:
        mn, mx = min(lex_raw), max(lex_raw)
        def norm_lex(x: float) -> float:
            if mx == mn:
                return 1.0 if x > 0 else 0.0
            return (x - mn) / (mx - mn)
        lex_norm = [norm_lex(x) for x in lex_raw]
    else:
        lex_norm = [0.0 for _ in candidates]

    for idx, (c, base_sim) in enumerate(zip(candidates, base_sims)):
        meta = c.get("metadata", {})
        content = meta.get("source_text", meta.get("source_text_preview", ""))
        c["_vec"] = norm_sim(base_sim)
        c["_lex"] = float(lex_norm[idx]) if idx < len(lex_norm) else 0.0
        c["_hybrid"] = alpha * c["_vec"] + (1 - alpha) * c["_lex"]
        c["rerank_score"] = c["_hybrid"]

    selected: List[Dict] = []
    remaining = sorted(candidates, key=lambda x: x.get("_hybrid", 0.0), reverse=True)
    content_tokens_cache: Dict[int, Set[str]] = {}

    def tokens_of(item: Dict) -> Set[str]:
        key = id(item)
        if key not in content_tokens_cache:
            meta = item.get("metadata", {})
            content = meta.get("source_text", meta.get("source_text_preview", ""))
            content_tokens_cache[key] = _tokenize(content)
        return content_tokens_cache[key]

    while remaining and len(selected) < top_k:
        best_item = None
        best_score = -1.0
        for cand in remaining[: candidate_k]:
            rel = cand.get("_hybrid", 0.0)
            if not selected:
                mmr = rel
            else:
                cand_tokens = tokens_of(cand)
                max_sim = 0.0
                for sel in selected:
                    sel_tokens = tokens_of(sel)
                    jacc = _jaccard(cand_tokens, sel_tokens)
                    if jacc > max_sim:
                        max_sim = jacc
                mmr = lambda_mmr * rel - (1 - lambda_mmr) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_item = cand

        if best_item is None:
            break
        selected.append(best_item)
        remaining.remove(best_item)

    selected.sort(key=lambda x: x.get("_hybrid", 0.0), reverse=True)
    return selected


def _tokenize(text: str) -> Set[str]:
    tokens: List[str] = []
    word = []
    for ch in (text or "").lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                tokens.append("".join(word))
                word = []
    if word:
        tokens.append("".join(word))
    return set(t for t in tokens if t and t not in _STOPWORDS)


def _tokenize_list(text: str) -> List[str]:
    tokens: List[str] = []
    word: List[str] = []
    for ch in (text or "").lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                tok = "".join(word)
                if tok and tok not in _STOPWORDS:
                    tokens.append(tok)
                word = []
    if word:
        tok = "".join(word)
        if tok and tok not in _STOPWORDS:
            tokens.append(tok)
    return tokens


_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "to", "was", "were", "with",
    "what", "which", "who", "whom", "when", "where", "why", "how",
}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    if union == 0:
        return 0.0
    return inter / union


def _base_relevance(result: Dict) -> float:

    if "distance" in result:
        try:
            d = float(result.get("distance", 0.0))
            return 1.0 / (1.0 + max(0.0, d))
        except Exception:
            pass
    try:
        return float(result.get("score", 0.0))
    except Exception:
        return 0.0


def _delete_vectors_batch(vector_keys: List[str], attempt: int = 1):
    try:
        s3_vectors_client.delete_vectors(
            vectorBucketName=settings.VECTOR_BUCKET_NAME,
            indexName=settings.INDEX_NAME,
            keys=vector_keys,
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")

        if code in {"Throttling", "ThrottlingException", "TooManyRequestsException", "RequestLimitExceeded"} and attempt <= settings.VECTOR_STORAGE_MAX_RETRIES:
            backoff = min(60.0, float(settings.VECTOR_STORAGE_BACKOFF_FACTOR) ** (attempt - 1))
            print(f"Throttling detected, retrying in {backoff} seconds (attempt {attempt})")
            time.sleep(backoff)
            _delete_vectors_batch(vector_keys, attempt + 1)
            return

        if code == "ResourceNotFoundException":
            print("Some vectors were already deleted or don't exist")
            return

        print(f"Error deleting vectors: {e}")
        raise
    except Exception as e:
        print(f"Error deleting vectors: {e}")
        raise


def delete_vectors_by_keys(vector_keys: List[str]):
    if not vector_keys:
        return

    print(f"Deleting {len(vector_keys)} vectors")

    batch_size = settings.VECTOR_PUT_BATCH_SIZE
    deleted_count = 0

    for i in range(0, len(vector_keys), batch_size):
        batch_keys = vector_keys[i:i + batch_size]
        try:
            _delete_vectors_batch(batch_keys)
            deleted_count += len(batch_keys)
            print(f"Deleted batch of {len(batch_keys)} vectors")
        except Exception as e:
            print(f"Error deleting vector batch: {e}")
            continue

    print(f"Successfully deleted {deleted_count} out of {len(vector_keys)} vectors")