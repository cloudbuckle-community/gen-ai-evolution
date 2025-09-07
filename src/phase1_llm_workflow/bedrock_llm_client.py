import json
import time
import boto3
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from botocore.exceptions import ClientError


@dataclass
class BedrockLLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    timestamp: float
    stop_reason: str = None


class BedrockLLMClient:
    """Amazon Bedrock LLM client for Anthropic Claude models"""

    def __init__(self,
                 model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
                 region_name: str = "us-east-1",
                 profile_name: str = None):
        self.model_id = model_id
        self.region_name = region_name
        self.profile_name = profile_name

        # Initialize Bedrock Runtime client
        session = boto3.Session(profile_name=profile_name)
        self.bedrock_runtime = session.client('bedrock-runtime', region_name=region_name)

        # Request tracking
        self.request_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def invoke(self,
               prompt: str,
               max_tokens: int = 1000,
               temperature: float = 0.7,
               top_p: float = 0.9,
               system_prompt: str = None) -> BedrockLLMResponse:
        """Invoke Claude model via Bedrock"""
        start_time = time.time()

        # Build request body for Anthropic Claude
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        # Add system prompt if provided
        if system_prompt:
            request_body["system"] = system_prompt

        try:
            # Make the API call
            response = self.bedrock_runtime.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType='application/json',
                accept='application/json'
            )

            # Parse response
            response_body = json.loads(response['body'].read())

            # Extract content and metadata
            content = response_body['content'][0]['text']
            input_tokens = response_body['usage']['input_tokens']
            output_tokens = response_body['usage']['output_tokens']
            stop_reason = response_body.get('stop_reason', 'end_turn')

            # Update tracking
            self.request_count += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            latency = (time.time() - start_time) * 1000

            return BedrockLLMResponse(
                content=content,
                model=self.model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                timestamp=time.time(),
                stop_reason=stop_reason
            )

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']

            # Handle specific Bedrock errors
            if error_code == 'ThrottlingException':
                raise Exception(f"Rate limit exceeded: {error_message}")
            elif error_code == 'ModelNotReadyException':
                raise Exception(f"Model not ready: {error_message}")
            elif error_code == 'ValidationException':
                raise Exception(f"Invalid request: {error_message}")
            else:
                raise Exception(f"Bedrock error ({error_code}): {error_message}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "request_count": self.request_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "model_id": self.model_id
        }