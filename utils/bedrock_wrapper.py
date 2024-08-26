from langchain_community.chat_models import BedrockChat

def init_bedrock_chat(model_id='anthropic.claude-3-sonnet-20240229-v1:0', region_name='us-west-2'):

    model_kwargs =  { 
        "max_tokens": 4096,
        "temperature": 0.0
    }
    bedrock_chat = BedrockChat(model_id=model_id, model_kwargs=model_kwargs, region_name=region_name)
    return bedrock_chat