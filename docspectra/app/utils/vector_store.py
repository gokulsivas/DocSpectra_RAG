import os
from pinecone import Pinecone, ServerlessSpec

# Init Pinecone client object
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index settings
index_name = "docspectra-index"
dimension = 1536
region = os.getenv("PINECONE_ENV", "us-east-1")

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=region
        )
    )

index = pc.Index(index_name)

def store_and_search_chunks(vectors: list[dict], query: str) -> list[str]:
    # Upsert chunks
    for i, item in enumerate(vectors):
        index.upsert(vectors=[
            {
                "id": f"id-{i}",
                "values": item["embedding"],
                "metadata": {"text": item["text"]}
            }
        ])

    # Embed query
    from app.utils.embedder import embed_chunks
    query_vector = embed_chunks([query])[0]["embedding"]

    # Search top 5
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True
    )

    return [match["metadata"]["text"] for match in results["matches"]]
