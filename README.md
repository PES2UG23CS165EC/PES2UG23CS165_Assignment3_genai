# Unit 3 Assignment: Advanced RAG System

- **HyDE Query Expansion** — rewrites the vague question into a richer hypothetical answer before searching
- **Hybrid Retrieval** — combines BM25 (keyword matching) + SBERT (semantic meaning) using RRF fusion
- **Cross-Encoder Re-Ranking** — re-scores the top retrieved docs using a more powerful model
- **LLM Generation** — generates a final answer grounded in the retrieved context

---

| Tool | Purpose |
|------|---------|
| `rank-bm25` | Keyword-based retrieval |
| `sentence-transformers` | Semantic (dense) retrieval + cross-encoder re-ranking |
| `Groq API (LLaMA 3.1)` | Query expansion (HyDE) + final answer generation |
| `Google Gemini API` | (optional) Alternative LLM for HyDE |

---

| Cell | Part | Description |
|------|------|-------------|
| Cell 1 | Setup | Install dependencies |
| Cell 2 | Setup | Imports and API keys |
| Cell 3 | Part 1 | Document corpus (13 AI/ML docs) |
| Cell 4 | Part 2 | HybridRetriever (BM25 + SBERT + RRF) |
| Cell 5 | Part 3 | Cross-Encoder re-ranker |
| Cell 6 | Part 4 | HyDE query expansion |
| Cell 7 | Part 5 | End-to-end `advanced_rag()` pipeline |
| Cell 8 | Part 5 | `naive_rag()` baseline |
| Cell 9 | Part 6 | Comparison experiment |
| Cell 10 | Part 6 | Results table + observations |

---

## Results Summary

| Query | Naïve RAG | Advanced RAG | Different? |
|-------|-----------|--------------|------------|
| "how do transformers encode meaning?" | attention mechanism... | attention mechanism... | No |
| "optimization techniques for training" | learning rate scheduling... | batch normalization... |  Yes |
| "what is the BLEU score used for?" | BLEU score... | BLEU score... | No |

**Key finding:** Advanced RAG outperforms naïve RAG on vague queries by using HyDE expansion to enrich the search. For already-specific or keyword-heavy queries, both pipelines perform equally well.

---

## Notes

- Cross-encoder scores can be negative — that is normal, higher = more relevant
- RRF scores are small numbers (e.g. 0.016) — use them for ranking only
- BM25 tokenizes on lowercase whitespace — preprocessing is handled internally
- `llama-3.1-8b-instant` is used instead of the deprecated `llama3-8b-8192`
