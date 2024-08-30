import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from utils.review_analyzer import (
    _split_df_to_docs, _analyze_review_by_lang, _analyze_review,
    _merge_review_by_lang, _merge_review, _write_analysis_report,
    _compare_analysis_result_by_lang, _compare_analysis_result,
    _init_data_by_lang, _init_data, analyze_data, analyze_data_by_lang,
    compare_target_data_by_lang, compare_target_data
)

class TestReviewAnalyzerSplitDFToDocs(unittest.TestCase):
    
    def setUp(self):
        self.sample_df = pd.DataFrame({
            'App Version Code': ['1.0', '2.0'],
            'Reviewer Language': ['en', 'fr'],
            'Device': ['device1', 'device2'],
            'Review Date': ['2023-01-01', '2023-01-02'],
            'Star Rating': [4, 3],
            'Review Title': ['Good', 'Okay'],
            'Review Text': ['Nice app', 'Could be better']
        })
        self.mock_bedrock_chat = MagicMock()
        self.mock_bedrock_chat.return_value = "Mocked response"

        
    def test_split_df_to_docs_basic(self):
        # 测试基本功能
        docs = _split_df_to_docs(self.sample_df)
        self.assertIsInstance(docs, list)
        self.assertTrue(len(docs) > 0)
        self.assertIsInstance(docs[0].page_content, str)

    def test_split_df_to_docs_content(self):
        # 测试文档内容
        docs = _split_df_to_docs(self.sample_df)
        content = docs[0].page_content
        self.assertIn('App Version Code', content)
        self.assertIn('Reviewer Language', content)
        self.assertIn('Device', content)
        self.assertIn('1.0', content)
        self.assertIn('en', content)
        self.assertIn('device1', content)

    def test_split_df_to_docs_empty_df(self):
        # 测试空DataFrame, 返回空列表
        empty_df = pd.DataFrame()
        docs = _split_df_to_docs(empty_df)
        self.assertEqual(len(docs), 0)
    
# class TestAnalyzeReviewByLang(unittest.TestCase):

#     def setUp(self):
#         self.mock_bedrock_chat = MagicMock()
#         self.sample_content = """
#         App Version Code,Reviewer Language,Device,Review Date,Star Rating,Review Title,Review Text
#         1.0,en,device1,2023-01-01,4,Good,Nice app
#         1.0,fr,device2,2023-01-02,2,Pas mal,Pourrait être mieux
#         """

#     @patch('utils.review_analyzer.PromptTemplate')
#     @patch('utils.review_analyzer.StrOutputParser')
#     def test_analyze_review_by_lang_basic(self, mock_str_output_parser, mock_prompt_template):
#         # Setup
#         mock_prompt_template.return_value = "Mocked prompt"
#         mock_str_output_parser.return_value.stream.return_value = ["<version='1.0' lang='en'>", "<issue>", "</issue>", "</version>"]
        
#         # Execute
#         result = _analyze_review_by_lang(self.sample_content, self.mock_bedrock_chat)
        
#         # Assert
#         self.assertIsInstance(result, str)
#         self.assertIn("<version='1.0' lang='en'>", result)
#         self.assertIn("<issue>", result)
#         self.assertIn("</issue>", result)
#         self.assertIn("</version>", result)

#     @patch('utils.review_analyzer.PromptTemplate')
#     @patch('utils.review_analyzer.StrOutputParser')
#     def test_analyze_review_by_lang_multiple_languages(self, mock_str_output_parser, mock_prompt_template):
#         # Setup
#         mock_prompt_template.return_value = "Mocked prompt"
#         mock_str_output_parser.return_value.stream.return_value = [
#             "<version='1.0' lang='en'>", "<issue>", "</issue>", "</version>",
#             "<version='1.0' lang='fr'>", "<issue>", "</issue>", "</version>"
#         ]
        
#         # Execute
#         result = _analyze_review_by_lang(self.sample_content, self.mock_bedrock_chat)
        
#         # Assert
#         self.assertIn("<version='1.0' lang='en'>", result)
#         self.assertIn("<version='1.0' lang='fr'>", result)

#     @patch('utils.review_analyzer.PromptTemplate')
#     @patch('utils.review_analyzer.StrOutputParser')
#     def test_analyze_review_by_lang_empty_content(self, mock_str_output_parser, mock_prompt_template):
#         # Setup
#         mock_prompt_template.return_value = "Mocked prompt"
#         mock_str_output_parser.return_value.stream.return_value = []
        
#         # Execute
#         result = _analyze_review_by_lang("", self.mock_bedrock_chat)
        
#         # Assert
#         self.assertEqual(result, "")

#     @patch('utils.review_analyzer.PromptTemplate')
#     @patch('utils.review_analyzer.StrOutputParser')
#     def test_analyze_review_by_lang_prompt_creation(self, mock_str_output_parser, mock_prompt_template):
#         # Execute
#         _analyze_review_by_lang(self.sample_content, self.mock_bedrock_chat)
#         
#         # Assert
#         mock_prompt_template.assert_called_once()
#         self.assertIn("document", mock_prompt_template.call_args[1]['input_variables'])

#     @patch('utils.review_analyzer.PromptTemplate')
#     @patch('utils.review_analyzer.StrOutputParser')
#     def test_analyze_review_by_lang_chain_creation(self, mock_str_output_parser, mock_prompt_template):
#         # Setup
#         mock_prompt_template.return_value = "Mocked prompt"
#         
#         # Execute
#         _analyze_review_by_lang(self.sample_content, self.mock_bedrock_chat)
#         
#         # Assert
#         self.mock_bedrock_chat.assert_called_once()
#         mock_str_output_parser.assert_called_once()


#     # @patch('utils.review_analyzer.StrOutputParser')
#     # def test_analyze_review_by_lang(self, mock_str_output_parser):
#     #     mock_str_output_parser.return_value.stream.return_value = ["Mocked", "stream", "response"]
#     #     result = _analyze_review_by_lang("Sample content", self.mock_bedrock_chat)
#     #     self.assertIsInstance(result, str)
#     #     self.assertEqual(result, "Mockedstreamresponse")

#     # @patch('utils.review_analyzer.StrOutputParser')
#     # def test_analyze_review(self, mock_str_output_parser):
#     #     mock_str_output_parser.return_value.stream.return_value = ["Mocked", "stream", "response"]
#     #     result = _analyze_review("Sample content", self.mock_bedrock_chat)
#     #     self.assertIsInstance(result, str)
#     #     self.assertEqual(result, "Mockedstreamresponse")

#     # @patch('utils.review_analyzer.StrOutputParser')
#     # def test_merge_review_by_lang(self, mock_str_output_parser):
#     #     mock_str_output_parser.return_value.stream.return_value = ["Merged", "review", "response"]
#     #     result = _merge_review_by_lang("Sample content", self.mock_bedrock_chat)
#     #     self.assertIsInstance(result, str)
#     #     self.assertEqual(result, "Mergedreviewresponse")

#     # @patch('utils.review_analyzer.StrOutputParser')
#     # def test_merge_review(self, mock_str_output_parser):
#     #     mock_str_output_parser.return_value.stream.return_value = ["Merged", "review", "response"]
#     #     result = _merge_review("Sample content", self.mock_bedrock_chat)
#     #     self.assertIsInstance(result, str)
#     #     self.assertEqual(result, "Mergedreviewresponse")

#     # @patch('utils.review_analyzer.StrOutputParser')
#     # def test_write_analysis_report(self, mock_str_output_parser):
#     #     # Assuming write_analysis_report is the function being tested
#     #     self.mock_bedrock_chat.return_value = "Analysisreportcontent"
#     #     result = _write_analysis_report("Sample content", self.mock_bedrock_chat)
#     #     self.assertEqual(result, "Analysisreportcontent")

#     # @patch('utils.review_analyzer.StrOutputParser')
#     # def test_compare_analysis_result_by_lang(self, mock_str_output_parser):
#     #     mock_str_output_parser.return_value.stream.return_value = ["Comparison", "result", "by lang"]
#     #     result = _compare_analysis_result_by_lang("Target data", "Baseline data", "1.0", "en", self.mock_bedrock_chat)
#     #     self.assertIsInstance(result, str)
#     #     self.assertEqual(result, "Comparisonresultby lang")

#     # @patch('utils.review_analyzer.StrOutputParser')
#     # def test_compare_analysis_result(self, mock_str_output_parser):
#     #     mock_str_output_parser.return_value.stream.return_value = ["Comparison", "result"]
#     #     result = _compare_analysis_result("Target data", "Baseline data", "1.0", self.mock_bedrock_chat)
#     #     self.assertIsInstance(result, str)
#     #     self.assertEqual(result, "Comparisonresult")

#     # @patch('utils.review_analyzer.st')
#     # def test_init_data_by_lang(self, mock_st):
#     #     result = _init_data_by_lang(self.sample_df)
#     #     self.assertIsInstance(result, dict)
#     #     self.assertEqual(len(result), 2)  # 'en' and 'fr'
#     #     self.assertIn('en', result)
#     #     self.assertIn('fr', result)

#     # @patch('utils.review_analyzer.st')
#     # def test_init_data(self, mock_st):
#     #     result = _init_data(self.sample_df)
#     #     self.assertIsInstance(result, dict)
#     #     self.assertEqual(len(result), 2)  # '1.0' and '2.0'
#     #     self.assertIn('1.0', result)
#     #     self.assertIn('2.0', result)

#     # @patch('utils.review_analyzer.st')
#     # @patch('utils.review_analyzer._init_data')
#     # @patch('utils.review_analyzer._analyze_review')
#     # @patch('utils.review_analyzer._merge_review')
#     # @patch('utils.review_analyzer._write_analysis_report')
#     # def test_analyze_data(self, mock_write, mock_merge, mock_analyze, mock_init, mock_st):
#     #     mock_init.return_value = {'1.0': [MagicMock()]}
#     #     mock_analyze.return_value = "Analyzed data"
#     #     mock_merge.return_value = "Merged data"
#     #     mock_write.return_value = "Written report"
#     
#     #     result = analyze_data(self.sample_df, self.mock_bedrock_chat)
#     
#     #     self.assertIsInstance(result, dict)
#     #     self.assertIn('1.0', result)
#     #     self.assertIn('xmldata', result['1.0'])
#     #     self.assertIn('report', result['1.0'])

#     # @patch('utils.review_analyzer.st')
#     # @patch('utils.review_analyzer._init_data_by_lang')
#     # @patch('utils.review_analyzer._analyze_review_by_lang')
#     # @patch('utils.review_analyzer._merge_review_by_lang')
#     # @patch('utils.review_analyzer._write_analysis_report')
#     # def test_analyze_data_by_lang(self, mock_write, mock_merge, mock_analyze, mock_init, mock_st):
#     #     mock_init.return_value = {'en': {'1.0': [MagicMock()]}}
#     #     mock_analyze.return_value = "Analyzed data"
#     #     mock_merge.return_value = "Merged data"
#     #     mock_write.return_value = "Written report"
#     
#     #     result = analyze_data_by_lang(self.sample_df, self.mock_bedrock_chat)
#     
#     #     self.assertIsInstance(result, dict)
#     #     self.assertIn('en', result)
#     #     self.assertIn('1.0', result['en'])
#     #     self.assertIn('xmldata', result['en']['1.0'])
#     #     self.assertIn('report', result['en']['1.0'])

#     # @patch('utils.review_analyzer.st')
#     # @patch('utils.review_analyzer._compare_analysis_result_by_lang')
#     # def test_compare_target_data_by_lang(self, mock_compare, mock_st):
#     #     mock_compare.return_value = "Comparison result"
#     #     analyze_result = {
#     #         'en': {
#     #             '1.0': {'xmldata': "<version='1.0' lang='en'><issue><category>Bug</category><count>5</count></issue></version>"},
#     #             '2.0': {'xmldata': "<version='2.0' lang='en'><issue><category>Bug</category><count>3</count></issue></version>"}
#     #         }
#     #     }
#     #     result = compare_target_data_by_lang("2.0", analyze_result, self.mock_bedrock_chat)
#     #     self.assertIsInstance(result, dict)
#     #     self.assertIn('en', result)
#     #     self.assertEqual(result['en'], "Comparison result")

#     # @patch('utils.review_analyzer.st')
#     # @patch('utils.review_analyzer._compare_analysis_result')
#     # def test_compare_target_data(self, mock_compare, mock_st):
#     #     mock_compare.return_value = "Comparison result"
#     #     analyze_result = {
#     #         '1.0': {'xmldata': "<version='1.0'><issue><category>Bug</category><count>5</count></issue></version>"},
#     #         '2.0': {'xmldata': "<version='2.0'><issue><category>Bug</category><count>3</count></issue></version>"}
#     #     }
#     #     result = compare_target_data("2.0", analyze_result, self.mock_bedrock_chat)
#     #     self.assertIsInstance(result, str)
#     #     self.assertEqual(result, "Comparison result")

class TestInitData(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            'App Version Code': ['1.0', '1.0', '2.0', '2.0'],
            'Reviewer Language': ['en', 'fr', 'en', 'de'],
            'Device': ['device1', 'device2', 'device3', 'device4'],
            'Review Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'Star Rating': [4, 3, 5, 2],
            'Review Title': ['Good', 'Okay', 'Great', 'Poor'],
            'Review Text': ['Nice app', 'Could be better', 'Love it', 'Needs improvement']
        })

    @patch('utils.review_analyzer.st')
    @patch('utils.review_analyzer._split_df_to_docs')
    def test_init_data_basic(self, mock_split_df_to_docs, mock_st):
        # Setup
        mock_split_df_to_docs.return_value = [MagicMock()]
        
        # Execute
        result = _init_data(self.sample_df)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Two unique versions: '1.0' and '2.0'
        self.assertIn('1.0', result)
        self.assertIn('2.0', result)
        self.assertEqual(mock_split_df_to_docs.call_count, 2)

    @patch('utils.review_analyzer.st')
    @patch('utils.review_analyzer._split_df_to_docs')
    def test_init_data_empty_df(self, mock_split_df_to_docs, mock_st):
        # Setup
        empty_df = pd.DataFrame()
        
        # Execute
        result = _init_data(empty_df)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)
        mock_split_df_to_docs.assert_not_called()

    @patch('utils.review_analyzer.st')
    @patch('utils.review_analyzer._split_df_to_docs')
    def test_init_data_single_version(self, mock_split_df_to_docs, mock_st):
        # Setup
        single_version_df = pd.DataFrame({
            'App Version Code': ['1.0', '1.0', '1.0'],
            'Reviewer Language': ['en', 'fr', 'de'],
            'Review Text': ['Text1', 'Text2', 'Text3']
        })
        mock_split_df_to_docs.return_value = [MagicMock()]
        
        # Execute
        result = _init_data(single_version_df)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertIn('1.0', result)
        mock_split_df_to_docs.assert_called_once()

    @patch('utils.review_analyzer.st')
    @patch('utils.review_analyzer._split_df_to_docs')
    def test_init_data_streamlit_calls(self, mock_split_df_to_docs, mock_st):
        # Setup
        mock_split_df_to_docs.return_value = [MagicMock()]
        
        # Execute
        _init_data(self.sample_df)
        
        # Assert
        mock_st.markdown.assert_called_once_with('**Start splitting data...**')
        self.assertEqual(mock_st.success.call_count, 2)  # One call for each version

class TestAnalyzeData(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            'App Version Code': ['1.0', '1.0', '2.0', '2.0'],
            'Reviewer Language': ['en', 'fr', 'en', 'de'],
            'Device': ['device1', 'device2', 'device3', 'device4'],
            'Review Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'Star Rating': [4, 3, 5, 2],
            'Review Title': ['Good', 'Okay', 'Great', 'Poor'],
            'Review Text': ['Nice app', 'Could be better', 'Love it', 'Needs improvement']
        })
        self.mock_bedrock_chat = MagicMock()
        self.mock_bedrock_chat.return_value = "Mocked LLM response"

    @patch('utils.review_analyzer._init_data')
    @patch('utils.review_analyzer._analyze_review')
    @patch('utils.review_analyzer._merge_review')
    @patch('utils.review_analyzer._write_analysis_report')
    @patch('utils.review_analyzer.st')
    def test_analyze_data_basic(self, mock_st, mock_write_report, mock_merge, mock_analyze, mock_init_data):
        # Setup
        mock_init_data.return_value = {'1.0': [MagicMock()], '2.0': [MagicMock()]}
        mock_analyze.return_value = "<mocked_analysis>"
        mock_merge.return_value = "<merged_analysis>"
        mock_write_report.return_value = "Mocked report"
        
        # Execute
        result = analyze_data(self.sample_df, self.mock_bedrock_chat)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)  # Two versions: '1.0' and '2.0'
        for version in ['1.0', '2.0']:
            self.assertIn(version, result)
            self.assertIn('xmldata', result[version])
            self.assertIn('report', result[version])
        mock_init_data.assert_called_once()
        self.assertEqual(mock_analyze.call_count, 2)  # Once for each version
        self.assertEqual(mock_merge.call_count, 2)
        self.assertEqual(mock_write_report.call_count, 2)

    @patch('utils.review_analyzer._init_data')
    @patch('utils.review_analyzer.st')
    def test_analyze_data_empty_df(self, mock_st, mock_init_data):
        # Setup
        empty_df = pd.DataFrame()
        mock_init_data.return_value = {}
        
        # Execute
        result = analyze_data(empty_df, self.mock_bedrock_chat)
        
        # Assert
        self.assertEqual(result, {})
        mock_init_data.assert_called_once_with(empty_df)

    @patch('utils.review_analyzer._init_data')
    @patch('utils.review_analyzer._analyze_review')
    @patch('utils.review_analyzer._merge_review')
    @patch('utils.review_analyzer._write_analysis_report')
    @patch('utils.review_analyzer.st')
    def test_analyze_data_single_version(self, mock_st, mock_write_report, mock_merge, mock_analyze, mock_init_data):
        # Setup
        single_version_df = pd.DataFrame({
            'App Version Code': ['1.0', '1.0', '1.0'],
            'Review Text': ['Text1', 'Text2', 'Text3']
        })
        mock_init_data.return_value = {'1.0': [MagicMock()]}
        mock_analyze.return_value = "<mocked_analysis>"
        mock_merge.return_value = "<merged_analysis>"
        mock_write_report.return_value = "Mocked report"
        
        # Execute
        result = analyze_data(single_version_df, self.mock_bedrock_chat)
        
        # Assert
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 1)
        self.assertIn('1.0', result)
        self.assertIn('xmldata', result['1.0'])
        self.assertIn('report', result['1.0'])
        mock_init_data.assert_called_once()
        mock_analyze.assert_called_once()
        mock_merge.assert_called_once()
        mock_write_report.assert_called_once()

    @patch('utils.review_analyzer._init_data')
    @patch('utils.review_analyzer._analyze_review')
    @patch('utils.review_analyzer._merge_review')
    @patch('utils.review_analyzer._write_analysis_report')
    @patch('utils.review_analyzer.st')
    def test_analyze_data_streamlit_calls(self, mock_st, mock_write_report, mock_merge, mock_analyze, mock_init_data):
        # Setup
        mock_init_data.return_value = {'1.0': [MagicMock()]}
        mock_analyze.return_value = "<mocked_analysis>"
        mock_merge.return_value = "<merged_analysis>"
        mock_write_report.return_value = "Mocked report"
        
        # Execute
        analyze_data(self.sample_df, self.mock_bedrock_chat)
        
        # Assert
        mock_st.markdown.assert_any_call('**Start analyzing data...**')
        self.assertGreater(mock_st.caption.call_count, 0)
        self.assertGreater(mock_st.success.call_count, 0)
        self.assertGreater(mock_st.divider.call_count, 0)


if __name__ == '__main__':
    unittest.main()