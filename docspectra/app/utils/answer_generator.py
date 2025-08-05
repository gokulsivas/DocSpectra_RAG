import boto3
import os

client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_DEFAULT_REGION"))

def generate_answer(query: str, chunks: list[str]) -> list:
    context = "\n\n".join(chunks)
    prompt = f"""You are an AI assistant helping users understand insurance policies.

Context:
{context}

Question: {query}
Answer:"""

    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "temperature": 0.7,
            "maxTokenCount": 300,
            "topP": 0.9,
            "stopSequences": []
        }
    }

    response = client.invoke_model(
        modelId=os.getenv("TITAN_MODEL_ID"),
        contentType="application/json",
        accept="application/json",
        body=bytes(str(body).replace("'", '"'), "utf-8"),
    )

    result = eval(response['body'].read().decode())
    return [{"clause": c.strip()} for c in result["results"][0]["outputText"].split("\n") if c.strip()]
