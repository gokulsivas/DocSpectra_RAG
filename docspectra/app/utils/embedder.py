import boto3
import os

model_id = os.getenv("TITAN_EMBED_MODEL")
region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

client = boto3.client("bedrock-runtime", region_name=region)

def embed_chunks(chunks: list[str]) -> list[dict]:
    embeddings = []
    for chunk in chunks:
        body = {
            "inputText": chunk
        }
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=bytes(str(body).replace("'", '"'), "utf-8"),
        )
        vector = eval(response['body'].read().decode())["embedding"]
        embeddings.append({"text": chunk, "embedding": vector})
    return embeddings
