import os
import yaml
from typing import Optional, Dict, Any, Generator
import boto3
from botocore.config import Config

config_yaml_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_yaml_path, 'r') as f:
    config_data = yaml.safe_load(f)
    
def list_translate_models():
    """
    Retrieve a list of bedrock model IDs from config.yaml

    Returns:
        list: A list of model IDs
    """
    return config_data['supportedmodel']

def list_bedrock_model_regions():
    """
    Retrieve a list of bedrock model regions from config.yaml

    Returns:
        list: A list of model regions
    """
    return config_data['model_region']


def get_bedrock_client(
    assumed_role: Optional[str] = None,
    region: Optional[str] = None,
    runtime: bool = True
) -> boto3.Session.client:
    """Create a boto3 client for Amazon Bedrock."""
    target_region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    
    session_kwargs: Dict[str, Any] = {"region_name": target_region}
    if profile_name := os.environ.get("AWS_PROFILE"):
        session_kwargs["profile_name"] = profile_name

    session = boto3.Session(**session_kwargs)

    if assumed_role:
        sts = session.client("sts")
        credentials = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )["Credentials"]
        client_kwargs = {
            "aws_access_key_id": credentials["AccessKeyId"],
            "aws_secret_access_key": credentials["SecretAccessKey"],
            "aws_session_token": credentials["SessionToken"],
        }
    else:
        client_kwargs = {}

    config = Config(
        region_name=target_region,
        retries={"max_attempts": 10, "mode": "standard"},
    )

    service_name = 'bedrock-runtime' if runtime else 'bedrock'
    return session.client(service_name=service_name, config=config, **client_kwargs)

# https://docs.anthropic.com/en/docs/about-claude/models#model-comparison

def invoke_bedrock_model(
    client: boto3.Session.client,
    model_id: str,
    system_prompt: str = '',
    prompt: str= '',
    show_details: bool = True,
    max_tokens: int = 4096,
    temperature: float = 0,
    top_p: float = 0.9
) -> str:
    """Invoke a Bedrock model."""
    try:
        response = None
        if system_prompt:
            response = client.converse(
                modelId=model_id,
                # system = [system_prompt],
                system = [{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "topP": top_p
                }
            )
        else:
            response = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "topP": top_p
                }
            )
        result = response['output']['message']['content'][0]['text']
        
        if show_details:
            metrics = response['metrics']
            usage = response['usage']
            result += (f"\n--- Latency: {metrics['latencyMs']}ms - "
                       f"Input tokens: {usage['inputTokens']} - "
                       f"Output tokens: {usage['outputTokens']} ---\n")
        
        return result
    except Exception as e:
        print(f"Error invoking model: {e}")
        return "Model invocation error"

def invoke_bedrock_model_stream(
    client: boto3.Session.client,
    model_id: str,
    system_prompt: str = '',
    prompt: str= '',
    max_tokens: int = 4096,
    temperature: float = 0,
    top_p: float = 0.9
) -> Generator[str, None, None]:
    """Invoke a Bedrock model with streaming response."""
    try:
        response = None
        if system_prompt:
            response = client.converse_stream(
                modelId=model_id,
                system = [{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "topP": top_p
                }
            )
        else:
            response = client.converse_stream(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "temperature": temperature,
                    "maxTokens": max_tokens,
                    "topP": top_p
                }
            )
            
        
        for event in response['stream']:
            if 'contentBlockDelta' in event:
                yield event['contentBlockDelta']['delta']['text']
    except Exception as e:
        print(f"Error in streaming invocation: {e}")
        yield "Streaming invocation error"
        
