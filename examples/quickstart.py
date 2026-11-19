"""Quickstart example demonstrating RAGPipeKit usage."""

from ragpipekit import RAGPipeline


def main():
    print("=== Initializing RAGPipeKit ===")
    rag = RAGPipeline(embedder="hash", chunk_size=200, store="memory")

    knowledge_base = [
        "LangGraph enables developers to build cyclic, stateful multi-agent systems with explicit flow control.",
        "FAISS is an efficient library developed by Meta for high-performance vector similarity search.",
        "Delta Lake brings ACID transactions and time travel capabilities to Apache Spark and big data pipelines.",
        "FastAPI provides fast, type-safe REST endpoint development leveraging modern Python typing.",
    ]

    print(f"Ingesting {len(knowledge_base)} documents...")
    chunks_indexed = rag.ingest(knowledge_base)
    print(f"Successfully indexed {chunks_indexed} text chunks into memory store.")

    user_query = "What library is best for vector search?"
    print(f"\nSearching for: '{user_query}'")
    results = rag.query(user_query, top_k=2)

    for i, res in enumerate(results, start=1):
        print(f"\nResult #{i} (Score: {res.score:.4f}):")
        print(f"Content: {res.content}")

    print("\nGenerated Prompt for Claude / LLMs:")
    print("-" * 50)
    print(rag.build_prompt(user_query))
    print("-" * 50)


if __name__ == "__main__":
    main()
