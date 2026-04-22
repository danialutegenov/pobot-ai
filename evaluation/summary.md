# Summary

This implementation builds a focused RAG assistant over fixed Labour Department sources related to migrant and employment support. The data pipeline downloads raw HTML/PDF snapshots, records a manifest for auditability, cleans and chunks text, and indexes chunk embeddings in FAISS. The Streamlit app retrieves relevant evidence for each query, then asks DeepSeek to answer with grounding constraints and source citations. Chinese questions are rewritten for English retrieval and answered in Chinese, while fallback behavior is triggered when confidence is low. A key limitation is that complex cross-ordinance questions may need broader retrieval context than top-k chunks provide.

