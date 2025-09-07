import os
import boto3
from typing import Dict, Any, Optional
from botocore.exceptions import ClientError


class BedrockConfig:
    """Configuration for Amazon Bedrock services"""

    def __init__(self, region_name: str = None, profile_name: str = None):
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        self.profile_name = profile_name or os.getenv("AWS_PROFILE")

        # Model configurations
        self.models = {
            "claude_sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude_haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
            "titan_embeddings": "amazon.titan-embed-text-v2:0",
            "titan_embeddings_v1": "amazon.titan-embed-text-v1"
        }

        # Default model settings
        self.default_llm_model = self.models["claude_sonnet"]
        self.default_embedding_model = self.models["titan_embeddings"]

        # Initialize Bedrock client
        self.bedrock_client = self._initialize_bedrock_client()
        self.bedrock_runtime = self._initialize_runtime_client()

    def _initialize_bedrock_client(self):
        """Initialize Bedrock client for model management"""
        try:
            session = boto3.Session(profile_name=self.profile_name)
            return session.client('bedrock', region_name=self.region_name)
        except Exception as e:
            print(f"Warning: Could not initialize Bedrock client: {e}")
            return None

    def _initialize_runtime_client(self):
        """Initialize Bedrock Runtime client for inference"""
        try:
            session = boto3.Session(profile_name=self.profile_name)
            return session.client('bedrock-runtime', region_name=self.region_name)
        except Exception as e:
            print(f"Warning: Could not initialize Bedrock Runtime client: {e}")
            return None

    def verify_model_access(self, model_id: str) -> bool:
        """Verify access to a specific model"""
        if not self.bedrock_client:
            return False

        try:
            response = self.bedrock_client.get_foundation_model(modelIdentifier=model_id)
            return True
        except ClientError as e:
            print(f"Model access verification failed for {model_id}: {e}")
            return False

    def list_available_models(self) -> list:
        """List all available foundation models"""
        if not self.bedrock_client:
            return []

        try:
            response = self.bedrock_client.list_foundation_models()
            return response.get('modelSummaries', [])
        except ClientError as e:
            print(f"Failed to list models: {e}")
            return []