import unittest
from unittest.mock import patch, MagicMock
from utils.bedrock_wrapper import init_bedrock_chat

class TestBedrockWrapper(unittest.TestCase):
    
    @patch('utils.bedrock_wrapper.BedrockChat')
    def test_init_bedrock_chat_default_params(self, mock_bedrock_chat):
        # Test with default parameters
        init_bedrock_chat()
        
        mock_bedrock_chat.assert_called_once_with(
            model_id='anthropic.claude-3-sonnet-20240229-v1:0',
            model_kwargs={
                "max_tokens": 4096,
                "temperature": 0.0
            },
            region_name='us-west-2'
        )

    @patch('utils.bedrock_wrapper.BedrockChat')
    def test_init_bedrock_chat_custom_params(self, mock_bedrock_chat):
        # Test with custom parameters
        custom_model_id = 'custom.model-id'
        custom_region = 'us-east-1'
        
        init_bedrock_chat(model_id=custom_model_id, region_name=custom_region)
        
        mock_bedrock_chat.assert_called_once_with(
            model_id=custom_model_id,
            model_kwargs={
                "max_tokens": 4096,
                "temperature": 0.0
            },
            region_name=custom_region
        )

    @patch('utils.bedrock_wrapper.BedrockChat')
    def test_init_bedrock_chat_return_value(self, mock_bedrock_chat):
        # Test that the function returns the BedrockChat instance
        mock_instance = MagicMock()
        mock_bedrock_chat.return_value = mock_instance
        
        result = init_bedrock_chat()
        
        
        self.assertEqual(result, mock_instance)

if __name__ == '__main__':
    unittest.main()
