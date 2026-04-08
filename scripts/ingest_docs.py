from pathlib import Path
from app.config import settings
from app.ingest import iter_document_paths, ingest_path
from app.embeddings import Embedder
from app.qdrant_store import QdrantStore


def main() -> None:
    docs_root = Path(settings.docs_path)
    paths = iter_document_paths(docs_root)
    if not paths:
        print(f"Inga dokument hittades i {docs_root}")
        return

    embedder = Embedder()
    probe = embedder.embed_query("test")
    store = QdrantStore(vector_size=len(probe))
    store.recreate_collection()

    total_chunks = 0
    for path in paths:
        chunks = ingest_path(path, docs_root)

        if chunks:
            print(f"{path.name}: {len(chunks)} chunks")
            print("  preview:", chunks[0].text[:250].replace("\n", " "))

        texts = [c.text for c in chunks if c.text.strip()]
        chunks = [c for c in chunks if c.text.strip()]

        if not chunks:
            continue

        vectors = embedder.embed_texts(texts)
        store.upsert_chunks(chunks, vectors)
        total_chunks += len(chunks)

    print(f"Klart. Indexerade {total_chunks} chunks.")


if __name__ == "__main__":
    main()