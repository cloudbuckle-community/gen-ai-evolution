import json
import numpy as np
import boto3
from typing import List, Dict, Any
from botocore.exceptions import ClientError


class BedrockEmbeddingsModel:
    """Amazon Bedrock embeddings using Titan models"""

    def __init__(self,
                 model_id: str = "amazon.titan-embed-text-v2:0",
                 region_name: str = "us-east-1",
                 profile_name: str = None):
        self.model_id = model_id
        self.region_name = region_name

        # Initialize Bedrock Runtime client
        session = boto3.Session(profile_name=profile_name)
        self.bedrock_runtime = session.client('bedrock-runtime', region_name=region_name)

        # Model-specific configurations
        self.model_configs = {
            "amazon.titan-embed-text-v2:0": {
                "dimension": 1024,
                "max_input_length": 8192,
                "supports_normalization": True
            },
            "amazon.titan-embed-text-v1": {
                "dimension": 1536,
                "max_input_length": 8192,
                "supports_normalization": False
            }
        }

        self.config = self.model_configs.get(model_id, self.model_configs["amazon.titan-embed-text-v2:0"])
        self.dimension = self.config["dimension"]

        # Usage tracking
        self.embedding_count = 0
        self.total_input_tokens = 0

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embeddings for a single text"""
        return self.embed_documents([text], normalize=normalize)[0]

    def embed_documents(self, texts: List[str], normalize: bool = True) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        embeddings = []

        for text in texts:
            # Truncate text if too long
            if len(text) > self.config["max_input_length"]:
                text = text[:self.config["max_input_length"]]

            embedding = self._get_embedding(text, normalize)
            embeddings.append(embedding)
            self.embedding_count += 1

        return embeddings

    def _get_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """Get embedding for a single text from Bedrock"""
        # Build request body for Titan embeddings
        request_body = {"inputText": text}

        # Add normalization for v2 models
        if self.config["supports_normalization"] and normalize:
            request_body["normalize"] = True

        try:
            # Make API call
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )

            # Parse response
            response_body = json.loads(response['body'].read())
            embedding_vector = response_body['embedding']

            # Track token usage if available
            if 'inputTextTokenCount' in response_body:
                self.total_input_tokens += response_body['inputTextTokenCount']

            # Convert to numpy array
            embedding = np.array(embedding_vector, dtype=np.float32)

            # Manual normalization for v1 models or if requested
            if not self.config["supports_normalization"] and normalize:
                embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise Exception(f"Bedrock embeddings error ({error_code}): {error_message}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "embedding_count": self.embedding_count,
            "total_input_tokens": self.total_input_tokens,
            "model_id": self.model_id,
            "dimension": self.dimension
        }