from typing import Dict, List, Optional
from app.services.embedding_service import create_embeddings_batch
from app.services.vector_service import query_vectors_hybrid, rerank_results
from app.core.config import settings
from app.services.llm_service import generate_answer


def ask_with_context(query: str, top_k: int = 5, use_reranking: bool = True, doc_ids: Optional[List[str]] = None) -> Dict:
    def _variations(q: str) -> List[str]:
        ql = q.lower()
        vars = [q]
        if any(w in ql for w in ['table', 'data', 'chart', 'figure']):
            vars.extend([f"table data about {q}", f"tabular information regarding {q}"])
        if any(w in ql for w in ['what is', 'define']):
            base = q.replace('what is', '').replace('define', '').strip()
            if base:
                vars.extend([f"definition of {base}", f"meaning of {base}"])
        return list(dict.fromkeys(vars))[:4]

    variations = _variations(query)
    embeddings = create_embeddings_batch(variations)

    all_results: List[Dict] = []
    for var_text, var_embedding in zip(variations, embeddings):
        var_results = query_vectors_hybrid(
            query_text=var_text,
            query_embedding=var_embedding,
            top_k=min(int(settings.RETRIEVAL_MAX_TOP_K), max(5, top_k)),
            candidate_k=min(int(settings.RETRIEVAL_MAX_CANDIDATES), max(int(settings.RETRIEVAL_CANDIDATE_K), top_k * 3)),
            alpha=float(settings.RETRIEVAL_ALPHA),
            lambda_mmr=float(settings.RETRIEVAL_LAMBDA_MMR),
            doc_ids=doc_ids,
        )
        if use_reranking:
            var_results = rerank_results(var_results, var_text)
        all_results.extend(var_results or [])

    if not all_results:
        return {
            "answer": "Insufficient information",
            "context": ""
        }

    dedup: Dict[str, dict] = {}
    for res in all_results:
        meta = res.get("metadata", {})
        # Prefer the vector key if present; fall back to filename+chunk_index.
        unique_key = str(res.get("key") or f"{meta.get('filename','')}|{meta.get('chunk_index','')}")
        score = res.get("rerank_score", res.get("score", 0.0))
        if unique_key not in dedup or score > dedup[unique_key].get("_score", 0.0):
            res_copy = dict(res)
            res_copy["_score"] = score
            dedup[unique_key] = res_copy

    merged_results = list(dedup.values())
    merged_results.sort(key=lambda r: r.get("_score", 0.0), reverse=True)
    results = merged_results[:top_k]

    context_parts: List[str] = []
    for i, result in enumerate(results[:5], 1):
        metadata = result.get("metadata", {})
        content = metadata.get("source_text", metadata.get("source_text_preview", ""))
        filename = metadata.get("filename", "")
        page_num = metadata.get("page_number", "")
        context_parts.append(f"[Source {i} - {filename}, Page {page_num}]\n{content}\n")
    context = "\n".join(context_parts)

    return {
        "context": context,
        "results": results
    }


def answer_with_context(query: str, top_k: int = 5, use_reranking: bool = True, doc_ids: Optional[List[str]] = None) -> str:
    ctx = ask_with_context(query, top_k=top_k, use_reranking=use_reranking, doc_ids=doc_ids)
    context = ctx.get("context", "")
    if not context.strip():
        return "Insufficient information"
    return generate_answer(query, context)


