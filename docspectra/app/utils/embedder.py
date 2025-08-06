import boto3, os, json
from botocore.exceptions import ClientError

model_id = os.getenv("TITAN_EMBED_MODEL")
region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

client = boto3.client("bedrock-runtime", region_name=region)

def embed_chunks(chunks: list[str]) -> list[dict]:
    embeddings = []
    for chunk in chunks:
        body = {"inputText": chunk}
        try:
            response = client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body).encode("utf-8"),
            )
        except ClientError as e:
            print("Bedrock invoke_model failed:", e)
            raise
        result = json.loads(response['body'].read().decode())
        embeddings.append({"text": chunk, "embedding": result["embedding"]})
    return embeddings
