# Mapping Kafka and Lispector

An exploratory framework for comparing Franz Kafka and Clarice Lispector across ten books in English translation (5,324 paragraphs total) using concept-erased sentence embeddings.

Raw embeddings segregate paragraphs by author and translator (linear classification distinguishes Kafka from Lispector with ~97% accuracy). This project scrubs that author signal to find cross-author paragraph pairs—termed _bridges_—that match on underlying theme rather than stylistic voice.

The interactive star map and scrollytelling article (`index.html`) are rendered with June, a custom design engine I made.

## Method

1. **Corpus & Chunking**: 10 books (5 Kafka, 5 Lispector) split into 5,324 passages bounded between 70 and 280 words (median 119 words).
2. **Theme-Instructed Embeddings**: Passages embedded via `Qwen/Qwen3-Embedding-0.6B` (with an explicit theme-extraction instruction) and `all-mpnet-base-v2`.
3. **Concept Erasure**:
   - **LEACE**: Closed-form projection removing linear author signal by aligning author centroids.
   - **CORAL**: Second-order alignment matching author covariance matrices in the top 256 dimensions.
4. **Hub-Corrected Matching**: Cross-Domain Similarity Local Scaling (CSLS) to suppress hub passages and find mutual nearest neighbours across writers.
5. **Validation & Ground Truth**: Evaluated on two independent English translations of _The Metamorphosis_ (Susan Bernofsky vs. Willa & Edwin Muir), where ground-truth paragraph alignment is known without models.

---
