from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st


def _split_df_to_docs(df, chunk_size=300000):
    """
    将DataFrame拆分为文本文档列表。

    Args:
        df (pandas.DataFrame): 要拆分的pandas DataFrame
        chunk_size (int, optional): 每个文档块的最大字符数，默认为300000

    Returns:
        list: 如果DataFrame为空，返回空列表，否则返回包含文本块的文档列表

    Example:
        Input:
            df = pd.DataFrame({
                'A': [1, 2, 3],
                'B': ['a', 'b', 'c']
            })
            chunk_size = 100

        Output:
            [Document(page_content="A  B\n1  a\n2  b\n3  c")]
    """
    # Check if the DataFrame is empty
    if df.empty:
        return []  # Return an empty list if the DataFrame is empty

    # 创建CharacterTextSplitter实例
    text_splitter = CharacterTextSplitter(
        separator="\n",  # 使用换行符作为分隔符
        chunk_size=chunk_size,  # 设置每个块的最大大小
        chunk_overlap=0,  # 块之间不重叠
        length_function=len,  # 使用len()函数计算长度
        is_separator_regex=False,  # 将分隔符视为普通字符串，而非正则表达式
    )

    # 将DataFrame转换为字符串，不包含索引
    df_string = df.to_string(index=False)
    
    # 使用text_splitter创建文档列表
    docs = text_splitter.create_documents([df_string])
    
    return docs

#region bedrock functions
# _analyze_review, 分析所有review，按照version group by后分析
# _merge_review, 将_analyze_review分析结果中同一version的结果，不同的批次合并
# _analyze_review_without_version, 分析所有review，不按照version group by，将所有review视为同一批次
# _merge_review_without_version, 将_analyze_review_without_version分析结果中不同的批次合并


# _analyze_review_by_lang
# _merge_review_by_lang
# _analyze_review_by_lang_without_version
# _merge_review_by_lang_without_version


# _write_analysis_report
# _compare_analysis_result
# _compare_analysis_result_by_lang
#endregion

# Analyze reviews by language
def _analyze_review_by_lang(content, bedrock_chat):
    # Define analysis prompt template
    """
    定义用于分析评论的提示模板。

    输入参数:
    - content (str): 包含评论数据的CSV格式字符串
    - bedrock_chat (function): 用于与Bedrock模型交互的函数

    返回值:
    - 分析结果，XML样式

    示例:
    输入:
        content = "App Version,Reviewer Language,Device,Review Date,Star Rating,Review Title,Review Text\n1.0,en,phone1,2023-01-01,2,Bad app,Crashes frequently"
        bedrock_chat = <function to interact with Bedrock model>

    输出:
        <version='version_a' lang='abc'>
            <issue>
                <category> issue x category</category>
                <count> how many reviews are in x category</count>
                <description>why player is dissatisfied for this issue category</description>
            </issue>
            <issue>
                <category> issue y category</category>
                <count> how many reviews are in y category</count>
                <description>why player is dissatisfied for this issue category</description>
            </issue>
        </version>
        <version='version_b' lang='abc'>
            <issue>
                <category> issue x category</category>
                <count> how many reviews are in x category</count>
                <description>why player is dissatisfied for this issue category</description>
            </issue>
            <issue>
                <category> issue y category</category>
                <count> how many reviews are in y category</count>
                <description>why player is dissatisfied for this issue category</description>
            </issue>
        </version>
    """
    analyze_prompt = PromptTemplate(
        template="""
        \n\nHuman: 

        You are an AI assistant trained to identify and categorize user negative reviews.
        You're specialized in many languages.
        You'll be provided with a batch of google play reviews in csv, the format is described in the <format> </format> tag.
        Your task is to identify and categorize customer negative reviews in <review></review> tag.
        You need to follow the instructions in <instructions></instructions> tag.

        <format>
        - Column 1, App Version: version code of the app.
        - Column 2, Code Reviewer Language: Language code for the reviewer.
        - Column 3, Device: Codename for the reviewer's device.
        - Column 4, Review Date: Date when the review was written.
        - Column 5, Star Rating: The star rating associated with the review, from 1 to 5.
        - Column 6, Review Title: The review title.
        - Column 7, Review Text: The review content.
        </format>

        <review> 
        {document}
        </review>

        <instructions>
        - review categories should be grouped by app version code and code reviewer language, using <version='xyz' lang='abc'> </version> tag
        - Identify and category negative reviews in the <category></category> tags, you can make categories on your own
        - Describe the issue in the <description></description> tag, explain why the player is dissatisfied
        - Your output must be a fully formatted xml file that intelligently contains the <version>, <issue>, <category>, <count>, <description> tags and no other tags.
        - You don't need to include the original review text
        </instructions>

        \n\nAssistant:
        <version='version_a' lang='abc'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </version>
        <version='version_b' lang='abc'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </version>
        """,
        input_variables=["document"]
    )

    # Create analysis chain
    insight_chain = analyze_prompt | bedrock_chat | StrOutputParser()
    
    # Stream process analysis results
    result_list=[]
    for chunk in insight_chain.stream({
        "document": {content},
    }):
        result_list.append(chunk)
        
    return ''.join(result_list)


def _analyze_review_by_lang_without_version(content, bedrock_chat):
    """
    Analyzes review data without version information using a language model.

    Args:
        content (str): A string containing review data in CSV format.
        bedrock_chat (function): A function that interfaces with the Amazon Bedrock language model.

    Returns:
        str: An XML string containing categorized negative reviews and their analysis.
        
    
    Example return:
    <issues lang='abc'>
        <issue>
            <category>App Stability</category>
            <count>1</count>
            <description>Users are experiencing frequent app crashes, leading to frustration and inability to use the app effectively.</description>
        </issue>
        <issue>
            <category>Login Issues</category>
            <count>1</count>
            <description>Users are unable to log in to the app, completely preventing them from accessing its features.</description>
        </issue>
        ...
    </issues>
    <issues lang='def'>
        <issue>
            <category>App Stability</category>
            <count>1</count>
            <description>Users are experiencing frequent app crashes, leading to frustration and inability to use the app effectively.</description>
        </issue>
        ...
    </issues>
    """
    analyze_prompt = PromptTemplate(
        template="""
        \n\nHuman: 

        You are an AI assistant trained to identify and categorize user negative reviews.
        You're specialized in many languages.
        You'll be provided with a batch of google play reviews in csv, the format is described in the <format> </format> tag.
        Your task is to identify and categorize customer negative reviews in <review></review> tag.
        You need to follow the instructions in <instructions></instructions> tag.
        
        <format>
        - Column 1, Code Reviewer Language: Language code for the reviewer.
        - Column 2, Device: Codename for the reviewer's device.
        - Column 3, Review Date: Date when the review was written.
        - Column 4, Star Rating: The star rating associated with the review, from 1 to 5.
        - Column 5, Review Title: The review title.
        - Column 6, Review Text: The review content.
        </format>

        <review> 
        {document}
        </review>
        
        <instructions>
        - Identify and category negative reviews in the <category></category> tags, you can make categories on your own
        - Describe the issue in the <description></description> tag, explain why the player is dissatisfied
        - Your output must be a fully formatted xml file that intelligently contains the <issues>, <issue>, <category>, <count>, <description> tags and no other tags.
        - You don't need to include the original review text
        </instructions>

        \n\nAssistant:
        <issues lang='abc'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </issues>
        <issues lang='def'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        ...
        </issues>
        """,
        input_variables=["document"]
    )

    # Create analysis chain
    insight_chain = analyze_prompt | bedrock_chat | StrOutputParser()
    
    # Stream process analysis results
    result_list=[]
    for chunk in insight_chain.stream({
        "document": {content},
    }):
        result_list.append(chunk)
        
    return ''.join(result_list)

def _analyze_review(content, bedrock_chat):
    # Define analysis prompt template
    analyze_prompt = PromptTemplate(
        template="""
        \n\nHuman: 

        You are an AI assistant trained to identify and categorize user negative reviews.
        You're specialized in many languages.
        You'll be provided with a batch of google play reviews in csv, the format is described in the <format> </format> tag.
        Your task is to identify and categorize customer negative reviews in <review></review> tag.
        You need to follow the instructions in <instructions></instructions> tag.

        <format>
        - Column 1, App Version: version code of the app.
        - Column 2, Code Reviewer Language: Language code for the reviewer.
        - Column 3, Device: Codename for the reviewer's device.
        - Column 4, Review Date: Date when the review was written.
        - Column 5, Star Rating: The star rating associated with the review, from 1 to 5.
        - Column 6, Review Title: The review title.
        - Column 7, Review Text: The review content.
        </format>

        <review> 
        {document}
        </review>

        <instructions>
        - review categories should be grouped by app version code, using <version='xyz'> </version> tag
        - Identify and category negative reviews in the <category></category> tags, you can make categories on your own
        - Describe the issue in the <description></description> tag, explain why the player is dissatisfied
        - Your output must be a fully formatted xml file that intelligently contains the <version>, <issue>, <category>, <count> tags and no other tags.
        - You don't need to include the original review text
        </instructions>

        \n\nAssistant:
        <version='version_a'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </version>
        <version='version_b'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </version>
        """,
        input_variables=["document"]
    )

    # Create analysis chain
    insight_chain = analyze_prompt | bedrock_chat | StrOutputParser()
    
    # Stream process analysis results
    result_list=[]
    for chunk in insight_chain.stream({
        "document": {content},
    }):
        result_list.append(chunk)
        
    return ''.join(result_list)

def _analyze_review_without_version(content, bedrock_chat):
    # Define analysis prompt template for reviews without version information
    """
    Analyzes review data without version information using a language model.

    Args:
        content (str): A string containing review data in CSV format.
        bedrock_chat (function): A function that interfaces with the Amazon Bedrock language model.

    Returns:
        str: An XML string containing categorized negative reviews and their analysis.

    Example input:
        content = "en,samsung-sm-g973f,2023-05-15,2,Disappointing,The app keeps crashing.\nfr,google-pixel-4a,2023-05-16,1,Terrible,Cannot log in at all."
        bedrock_chat = <function that interfaces with Amazon Bedrock>

    Example output:
        <issues>
        <issue>
        <category>App Stability</category>
        <count>1</count>
        <description>Users are experiencing frequent app crashes, leading to frustration and inability to use the app effectively.</description>
        </issue>
        <issue>
        <category>Login Issues</category>
        <count>1</count>
        <description>Users are unable to log in to the app, completely preventing them from accessing its features.</description>
        </issue>
        </issues>
    """
    analyze_prompt = PromptTemplate(
        template="""
        \n\nHuman: 

        You are an AI assistant trained to identify and categorize user negative reviews.
        You're specialized in many languages.
        You'll be provided with a batch of google play reviews in csv, the format is described in the <format> </format> tag.
        Your task is to identify and categorize customer negative reviews in <review></review> tag.
        You need to follow the instructions in <instructions></instructions> tag.

        <format>
        - Column 1, Code Reviewer Language: Language code for the reviewer.
        - Column 2, Device: Codename for the reviewer's device.
        - Column 3, Review Date: Date when the review was written.
        - Column 4, Star Rating: The star rating associated with the review, from 1 to 5.
        - Column 5, Review Title: The review title.
        - Column 6, Review Text: The review content.
        </format>

        <review> 
        {document}
        </review>

        <instructions>
        - Identify and category negative reviews in the <category></category> tags, you can make categories on your own
        - Describe the issue in the <description></description> tag, explain why the player is dissatisfied
        - Your output must be a fully formatted xml file that intelligently contains the <issues>, <issue>, <category>, <count> tags and no other tags.
        - You don't need to include the original review text
        </instructions>

        \n\nAssistant:
        <issues>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </issues>
        """,
        input_variables=["document"]
    )

    # Create analysis chain
    insight_chain = analyze_prompt | bedrock_chat | StrOutputParser()
    
    # Stream process analysis results
    result_list=[]
    for chunk in insight_chain.stream({
        "document": {content},
    }):
        result_list.append(chunk)
        
    return ''.join(result_list)

def _analyze_review_without_version_by_lang(content, bedrock_chat):
    pass


# Merge review analysis results classified by language
def _merge_review_by_lang(content, bedrock_chat):
    # Define merge prompt template
    merge_prompt = PromptTemplate(
    template="""
        You are an AI assistant. 
        You're specialized in many languages.
        You'll be provided with a batch of review issues categoried by version in xml format, your task is to merge the issues with the same or similar meaning in <content></content> tag. 
        You need to follow the instructions in <instructions></instructions> tags.
        
        <content> {reviews} </content>

        <instructions>
        - Merge the issues with the same or similar meaning
        - You must update the <count></count> tag of the merged issue to the sum of the counts of the merged issues
        </instructions>

        \n\nAssistant:
        <version='version_a' lang='abc'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </version>
        <version='version_b' lang='abc'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </version>
        """,
        input_variables=["reviews"]
    )

    # Create merge chain
    merge_chain = merge_prompt | bedrock_chat | StrOutputParser()
    # Stream process merge results
    result_list=[]
    for chunk in merge_chain.stream({
        "reviews": {content},
    }):
        result_list.append(chunk)
        
    return ''.join(result_list)

def _merge_review_without_version_by_lang(content, bedrock_chat):
    merge_prompt = PromptTemplate(
    template="""
        You are an AI assistant. 
        You're specialized in many languages.
        You'll be provided with a batch of review issues categoried by version in xml format, your task is to merge the issues with the same or similar meaning in <content></content> tag. 
        You need to follow the instructions in <instructions></instructions> tags.
        
        <content> {reviews} </content>

        <instructions>
        - Merge the issues with the same or similar meaning
        - You must update the <count></count> tag of the merged issue to the sum of the counts of the merged issues
        </instructions>

        \n\nAssistant:
        <issues>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </issues>
    """,
        input_variables=["reviews"]
    )

    # Create merge chain
    merge_chain = merge_prompt | bedrock_chat | StrOutputParser()
    # Stream process merge results
    result_list=[]
    for chunk in merge_chain.stream({
        "reviews": {content},
    }):
        result_list.append(chunk)
        
    return ''.join(result_list)

# Merge review analysis results (not classified by language)
def _merge_review(content, bedrock_chat):
    # Define merge prompt template
    merge_prompt = PromptTemplate(
    template="""
        You are an AI assistant. 
        You're specialized in many languages.
        You'll be provided with a batch of review issues categoried by version in xml format, your task is to merge the issues with the same or similar meaning in <content></content> tag. 
        You need to follow the instructions in <instructions></instructions> tags.
        
        <content> {reviews} </content>

        <instructions>
        - Merge the issues with the same or similar meaning
        - You must update the <count></count> tag of the merged issue to the sum of the counts of the merged issues
        </instructions>

        \n\nAssistant:
        <version='version_a'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </version>
        <version='version_b'>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </version>
        """,
        input_variables=["reviews"]
    )

    # Create merge chain
    merge_chain = merge_prompt | bedrock_chat | StrOutputParser()
    # Stream process merge results
    result_list=[]
    for chunk in merge_chain.stream({
        "reviews": {content},
    }):
        result_list.append(chunk)
        
    return ''.join(result_list)


def _merge_review_without_version(content, bedrock_chat):
    merge_prompt = PromptTemplate(
    template="""
        You are an AI assistant. 
        You're specialized in many languages.
        You'll be provided with a batch of review issues categoried by version in xml format, your task is to merge the issues with the same or similar meaning in <content></content> tag. 
        You need to follow the instructions in <instructions></instructions> tags.
        
        <content> {reviews} </content>

        <instructions>
        - Merge the issues with the same or similar meaning
        - You must update the <count></count> tag of the merged issue to the sum of the counts of the merged issues
        </instructions>

        \n\nAssistant:
        <issues>
        <issue>
        <category> issue x category</category>
        <count> how many reviews are in x category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        <issue>
        <category> issue y category</category>
        <count> how many reviews are in y category</count>
        <description>why player is dissatisfied for this issue category</description>
        </issue>
        </issues>
    """,
        input_variables=["reviews"]
    )

    # Create merge chain
    merge_chain = merge_prompt | bedrock_chat | StrOutputParser()
    # Stream process merge results
    result_list=[]
    for chunk in merge_chain.stream({
        "reviews": {content},
    }):
        result_list.append(chunk)
        
    return ''.join(result_list)

# Generate analysis report
def _write_analysis_report(content, bedrock):
    # Define report generation prompt template
    writing_prompt = PromptTemplate(
    template="""
        You're an AI assistant who's proficient in multiple languages and good at writing.
        You'll be provided with a batch of review issues categoried by version in xml format, your task is to analyze the material provided in the <content></content> tag and then write an analysis in markdown format. 
        You need to follow the instructions in <instructions></instructions> tags.

        <content> {reviews} </content>

        <instructions>
        - The analysis must be in markdown format, you can use bold, but you must not use heading.
        - The analysis must be in Simplified Chinese, you must not use Traditional Chinese.
        - The analysis must contain three sections: Summary, Key Issues Analysis and Conclusion.
        - You must calculate the total number of counts by adding up the counts of each issue, and you must include this total number of counts in the Summary, outputting as: "Total number of comments with non-empty and meaningful content is xx".
        - For each issue, you must analysis it in bullet format, and you must include what the count of the Issue is, what the percentage of the Issue count is to the total of all Issue counts, outputting as: "Count:xxx, Percentage: yy.yy%".
        - For each issue analysis, you must explain in detail why and what the player is dissatisfied with in a new line.
        </instructions>

        \n\nAssistant:
        """,
        input_variables=["reviews"]
    )

    # Create report generation chain
    writing_chain = writing_prompt | bedrock | StrOutputParser()
    # Stream process report generation results
    result_list = []
    for chunk in writing_chain.stream({
        "reviews": {content},
    }):
        if isinstance(chunk, str):
            result_list.append(chunk)
        else:
            # Handle non-string responses, e.g., log a warning or skip
            logging.warning(f"Unexpected response type: {type(chunk)}")

    return ''.join(result_list)

# Compare analysis results classified by language
def _compare_analysis_result_by_lang(target_data, baseline_data, target_version_no, lang, bedrock):
    # Define comparison prompt template
    compare_prompt = PromptTemplate(
    template="""
        You are an AI assistant who's proficient in multiple languages.
        You'll be provided with the analysis of a game app review issues of target version in xml format in <target></target> tag,
        and will be provided with several other version's analysis as baseline in xml format in <baseline></baseline> tag.
        Your task is to analyze the review issues, compare the target version  review issues with the baseline version's, find out what new problems have arisen in the target version, 
        find out what problems have become worse in the target version. 
        You need to follow the instructions in <instructions></instructions> tags.
        
        <target> {target_data} </target>
        <baseline> {baseline_data} </baseline>

        <instructions>
        - You must output a full analysis report in markdown format in Simplified Chinese, you must not use Traditional Chinese.
        - Your report must be in markdown format, you can use bold, but you must not use heading.
        - Your report must not include any xml tag.
        - You must explain your ideas in detail and provide data to support them.
        - Your report must include a detailed description about the new found problem(s).
        - Your report must include a detailed description about the problem(s) that have become more serious.
        - <baseline></baseline> tag may contain more than one version of the review issue analysis, you must analyze each version of the review issue analysis
        - The report header must include target_version_number and lang, in the format of: **Comparative Report for Version {target_version_no}, Language Code {lang}**
        </instructions>

        \n\nAssistant:
        """,
        input_variables=["target_data","baseline_data", "target_version_no", "lang"]
    )

    # Create comparison chain
    compare_chain = compare_prompt | bedrock | StrOutputParser()
    
    # Stream process comparison results
    compare_list=[]
    for chunk in compare_chain.stream({
            "target_data": target_data,
            "baseline_data": baseline_data,
            "target_version_no": target_version_no,
            "lang": lang
        }):
        compare_list.append(chunk)

    return ''.join(compare_list)

# Compare analysis results (not classified by language)
def _compare_analysis_result(target_data, baseline_data, target_version_no, bedrock):
    # Define comparison prompt template
    compare_prompt = PromptTemplate(
    template="""
        You are an AI assistant who's proficient in multiple languages.
        You'll be provided with the analysis of a game app review issues of target version in xml format in <target></target> tag,
        and will be provided with several other version's analysis as baseline in xml format in <baseline></baseline> tag.
        Your task is to analyze the review issues, compare the target version  review issues with the baseline version's, find out what new problems have arisen in the target version, 
        find out what problems have become worse in the target version. 
        You need to follow the instructions in <instructions></instructions> tags.
        
        <target> {target_data} </target>
        <baseline> {baseline_data} </baseline>

        <instructions>
        - You must output a full analysis report in markdown format in Simplified Chinese, you must not use Traditional Chinese.
        - Your report must be in markdown format, you can use bold, but you must not use heading.
        - Your report must not include any xml tag.
        - You must explain your ideas in detail and provide data to support them.
        - Your report must include a detailed description about the new found problem(s).
        - Your report must include a detailed description about the problem(s) that have become more serious.
        - <baseline></baseline> tag may contain more than one version of the review issue analysis, you must analyze each version of the review issue analysis
        - The report header must include target_version_number and lang, in the format of: **Comparative Report for Version {target_version_no}**
        </instructions>

        \n\nAssistant:
        """,
        input_variables=["target_data","baseline_data", "target_version_no"]
    )

    # Create comparison chain
    compare_chain = compare_prompt | bedrock | StrOutputParser()
    
    # Stream process comparison results
    compare_list=[]
    for chunk in compare_chain.stream({
            "target_data": target_data,
            "baseline_data": baseline_data,
            "target_version_no": target_version_no
        }):
        compare_list.append(chunk)

    return ''.join(compare_list)

# Initialize data classified by language
def _init_data_by_lang(data):
    """
    Initializes and preprocesses the review data for analysis, classified by language and app version.

    Args:
        data (pandas.DataFrame): A DataFrame containing review information.
            Expected columns: 'Reviewer Language', 'App Version Code', and other review-related columns.

    Returns:
        dict: A nested dictionary where the first level keys are language codes, 
              the second level keys are app version codes, 
              and values are lists of document chunks.
            Each chunk is a manageable subset of the review data for that language and version.

    Note:
        This function uses Streamlit (st) to display progress messages during execution.

    Example:
        Input:
            data = pd.DataFrame({
                'Reviewer Language': ['en', 'fr', 'en', 'fr'],
                'App Version Code': ['1.0', '1.0', '2.0', '2.0'],
                'Review Text': ['Good app', 'Pas mal', 'Great update', 'Très bien']
            })

        Output:
            {
                'en': {
                    '1.0': [Document(page_content="Reviewer Language App Version Code Review Text\nen 1.0 Good app")],
                    '2.0': [Document(page_content="Reviewer Language App Version Code Review Text\nen 2.0 Great update")]
                },
                'fr': {
                    '1.0': [Document(page_content="Reviewer Language App Version Code Review Text\nfr 1.0 Pas mal")],
                    '2.0': [Document(page_content="Reviewer Language App Version Code Review Text\nfr 2.0 Très bien")]
                }
            }
    """
    st.markdown('''**Start splitting data...**''')
    raw = {}
    for lang in data['Reviewer Language'].unique():
        raw[lang] = {}
        for version in data['App Version Code'].unique():
            target_data = data[(data['Reviewer Language'] == lang) & (data['App Version Code'] == version)]
            docs = _split_df_to_docs(target_data)
            raw[lang][version] = docs
            st.success(f"Data split: language {lang}, version:{version}, total {len(docs)} batches", icon="✅")
    return raw


def _init_data_by_lang_without_version(data):
    """
    Initializes and preprocesses the review data for analysis, classified by language.

    Args:
        data (pandas.DataFrame): A DataFrame containing review information.
            Expected columns: 'Reviewer Language', and other review-related columns.

    Returns:
        dict: A dictionary where keys are language codes and values are lists of document chunks.
            Each chunk is a manageable subset of the review data for that language.

    Example:
        Input:
            data = pd.DataFrame({
                'Reviewer Language': ['en', 'fr', 'en', 'fr'],
                'Review Text': ['Good app', 'Pas mal', 'Great update', 'Très bien']
            })

        Output:
            {
                'en': [Document(page_content="Reviewer Language Review Text\nen Good app\nen Great update")],
                'fr': [Document(page_content="Reviewer Language Review Text\nfr Pas mal\nfr Très bien")]
            }
    """
    st.markdown('''**Start splitting data...**''')
    raw = {}
    for lang in data['Reviewer Language'].unique():
        raw[lang] = {}
        target_data = data[data['Reviewer Language'] == lang]
        docs = _split_df_to_docs(target_data)
        raw[lang] = docs
        st.success(f"Data split: language {lang}, total {len(docs)} batches", icon="✅")
    return raw



def _init_data(data):
    """
    Initializes and preprocesses the review data for analysis.

    Args:
        data (pandas.DataFrame): A DataFrame containing review information.
            Expected columns: 'App Version Code', and other review-related columns.

    Returns:
        dict: A dictionary where keys are app version codes and values are lists of document chunks.
            Each chunk is a manageable subset of the review data for that version.

    Note:
        This function uses Streamlit (st) to display progress messages during execution.

    Example input data:
    data = pd.DataFrame({
        'App Version Code': ['1.0', '1.0', '2.0', '2.0'],
        'Reviewer Language': ['en', 'fr', 'en', 'fr'],
        'Device': ['phone1', 'phone2', 'tablet1', 'tablet2'],
        'Review Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Star Rating': [4, 3, 5, 2],
        'Review Title': ['Good app', 'Moyenne', 'Great update', 'Besoin d'amélioration'],
        'Review Text': ['Works well', 'Pas mal mais peut mieux faire', 'Love the new features', 'Trop de bugs']
    })

    Example output of raw:
    raw = {
        '1.0': [Document(page_content="App Version Code Reviewer Language Device Review Date Star Rating Review Title Review Text\n1.0 en phone1 2023-01-01 4 Good app Works well\n1.0 fr phone2 2023-01-02 3 Moyenne Pas mal mais peut mieux faire")],
        '2.0': [Document(page_content="App Version Code Reviewer Language Device Review Date Star Rating Review Title Review Text\n2.0 en tablet1 2023-01-03 5 Great update Love the new features\n2.0 fr tablet2 2023-01-04 2 Besoin d'amélioration Trop de bugs")]
    }
    """

    # Display a message indicating the start of data splitting process
    st.markdown('''**Start splitting data...**''')
    
    # Initialize an empty dictionary to store processed data
    raw = {}
    
    # Iterate through unique app version codes in the data
    if not data.empty:
        for version in data['App Version Code'].unique():
            # Filter data for the current version
            target_data = data[data['App Version Code'] == version]
            
            # Split the filtered data into manageable chunks
            docs = _split_df_to_docs(target_data)
            
            # Store the split data in the raw dictionary, keyed by version
            raw[version] = docs
            
            # Display a success message with details about the split data
            st.success(f"Data split completed: version:{version} total {len(target_data)} items, split into {len(docs)} batches for processing", icon="✅")
        
    # Return the processed data dictionary
    return raw

def _init_data_without_version(data):
    """
    Initializes and preprocesses the review data for analysis without version information.

    Args:
        data (pandas.DataFrame): A DataFrame containing review information.
    
    Returns:
        list: A list of document chunks, where each chunk is a manageable subset of the review data.

    Note:
        This function uses Streamlit (st) to display progress messages during execution.

    Example input:
        data = pd.DataFrame({
            'Reviewer Language': ['en', 'fr', 'en', 'de'],
            'Review Text': ['Great app', 'Needs improvement', 'Love it', 'Bug in latest version'],
            'Star Rating': [5, 2, 4, 3]
        })
    Example output:
        raw = [
            Document(page_content="Reviewer Language Review Text Star Rating\nen Great app 5\nfr Needs improvement 2"),
            Document(page_content="en Love it 4\nde Bug in latest version 3")
        ]
    """
    st.markdown('''**Start splitting data...**''')
    raw=[]
    if not data.empty:
        raw = _split_df_to_docs(data)
    st.success(f"Data split: total {len(raw)} batches",icon="✅")   
    return raw
    
# Analyze data (main function)
@st.cache_data
def analyze_data(data, _bedrock_chat):
    """
    Analyzes review data using a language model provided by Amazon Bedrock.

    Args:
        data (pandas.DataFrame): A DataFrame containing review information.
            Expected columns: 'App Version Code', 'Review Text', and other review-related columns.
        _bedrock_chat (function): A function that interfaces with the Amazon Bedrock language model.

    Returns:
        dict: A dictionary where keys are app version codes and values are dictionaries containing:
            - "xmldata": The merged XML data from the analysis.
            - "report": The generated report in markdown format.

    Note:
        This function uses Streamlit (st) to display progress messages and reports during execution.
        It also uses st.cache_data for caching the results.

    Example input:
        data = pd.DataFrame({
            'App Version Code': ['1.0', '1.0', '2.0', '2.0'],
            'Reviewer Language': ['en', 'fr', 'en', 'de'],
            'Review Text': ['Great app', 'Needs improvement', 'Love it', 'Bug in latest version'],
            'Star Rating': [5, 2, 4, 3]
        })
        _bedrock_chat = xxx

    Example output:
        {
            '1.0': {
                'xmldata': '<version="1.0"><issue><category>Positive Feedback</category><count>1</count>...</issue></version>',
                'report': '## Summary\nTotal number of comments: 2\n\n## Key Issues Analysis\n- Positive Feedback (50%): Users appreciate the app...'
            },
            '2.0': {
                'xmldata': '<version="2.0"><issue><category>Bug Report</category><count>1</count>...</issue></version>',
                'report': '## Summary\nTotal number of comments: 2\n\n## Key Issues Analysis\n- Bug Report (50%): Users reported a bug in the latest version...'
            }
        }
    """
    # Initialize data
    raw = _init_data(data)
    st.markdown('''**Start analyzing data...**''')
    analyze_result = {}
    
    # Iterate through each version of data for analysis
    for version in raw:
        docs = raw[version]
        analyze_result[version]={}
        st.caption(f'''Start analyzing dataset version {version}, total {len(docs)} batches''')
        
        # Analyze data in batches
        chunk_result= []
        for i, doc in enumerate(docs, start=1):
            st.caption(f'''- Analyzing batch {i}, {len(doc.page_content.split('\n'))} items...''')
            chunk_result.append(_analyze_review(doc.page_content, _bedrock_chat))
        st.success(f"Analysis of dataset version {version} completed", icon="✅")

        # Merge analysis results
        if len(docs) > 1:
            st.caption(f'''Start merging dataset version {version}''')
            analyze_result[version]["xmldata"]=_merge_review(''.join(chunk_result), _bedrock_chat)
            st.success(f"Merging dataset version {version} completed",icon="✅")
        else:
            analyze_result[version]["xmldata"] = chunk_result[0]
        
        # Generate report
        st.caption(f'''Start translating and writing report: version {version}''')
        st.divider()
        analyze_result[version]["report"] = _write_analysis_report(analyze_result[version]["xmldata"], _bedrock_chat)
        st.success(f'''Report completed: version {version}''',icon="✅")
        st.markdown(analyze_result[version]["report"])
        st.divider()
    return analyze_result

@st.cache_data
def analyze_data_without_version(data, _bedrock_chat):
    """
    Analyzes review data without version information using a language model provided by Amazon Bedrock.

    Args:
        data (pandas.DataFrame): A DataFrame containing review information.
            Expected columns: 'Review Text', and other review-related columns.

    Returns:
        dict: A dictionary containing:
            - "xmldata": The merged XML data from the analysis.
            - "report": The generated report in markdown format.

        Example:
            Input:
                data = pd.DataFrame({
                    'Review Text': ['Great app!', 'Crashes often'],
                    'Star Rating': [5, 2]
                })
            
            Output:
                {
                    'xmldata': '<issues><issue><category>Positive Feedback</category><count>1</count>...</issue><issue><category>App Stability</category><count>1</count>...</issue></issues>',
                    'report': '## Summary\nTotal number of comments: 2\n\n## Key Issues Analysis\n- Positive Feedback (50%): Users appreciate the app...\n- App Stability (50%): Some users report frequent crashes...'
                }
            - "xmldata": The merged XML data from the analysis.
            - "report": The generated report in markdown format.

    Note:
        This function uses Streamlit (st) to display progress messages and reports during execution.
        It also uses st.cache_data for caching the results.
    """
    
    data_removed_version = data.drop(columns=['App Version Code'])
    raw = _init_data_without_version(data_removed_version) # raw is a list of docs
    analyze_result = {}
    st.markdown('''**Start analyzing data...**''')
    chunk_result= []
    for i, doc in enumerate(raw, start=1):
        st.caption(f'''- Analyzing batch {i}, {len(doc.page_content.split('\n'))} items...''')
        chunk_result.append(_analyze_review_without_version(doc.page_content, _bedrock_chat))
    st.success(f"Analysis completed", icon="✅")
    
    if len(chunk_result) > 1:
        st.caption(f'''Start merging dataset''')
        analyze_result["xmldata"] = _merge_review_without_version(''.join(chunk_result), _bedrock_chat)
        st.success(f"Merging dataset completed",icon="✅")
    else:
        analyze_result["xmldata"] = chunk_result[0]
    
    st.caption(f'''Start translating and writing report''')
    analyze_result["report"] = _write_analysis_report(analyze_result["xmldata"], _bedrock_chat)
    st.success(f"Report completed",icon="✅")
    st.markdown(analyze_result["report"])
    return analyze_result
    


# Analyze data by language (main function)
@st.cache_data
def analyze_data_by_lang(data, _bedrock_chat):
    """
    Analyzes review data by language and version using a language model provided by Amazon Bedrock.

    Args:
        data (pandas.DataFrame): A DataFrame containing review information.
            Expected columns: 'App Version Code', 'Reviewer Language', 'Review Text', and other review-related columns.
        _bedrock_chat (function): A function that interfaces with the Amazon Bedrock language model.

    Returns:
        dict: A nested dictionary containing analysis results for each language and version.
            Structure: {language: {version: {'xmldata': str, 'report': str}}}

    Example:
        Input:
            data = pd.DataFrame({
                'App Version Code': ['1.0', '2.0', '1.0'],
                'Reviewer Language': ['en', 'fr', 'en'],
                'Review Text': ['Great app', 'Needs improvement', 'Love it']
            })
            _bedrock_chat = some_bedrock_chat_function

        Output:
            {
                'en': {
                    '1.0': {
                        'xmldata': '<version="1.0">...</version>',
                        'report': '**Analysis Report for Version 1.0, Language: English**\n\n...'
                    },
                    '2.0': {...}
                },
                'fr': {
                    '2.0': {...}
                }
            }

    Note:
        This function uses Streamlit (st) to display progress messages and reports during execution.
        It also uses st.cache_data for caching the results.
    """
    # Initialize data
    raw = _init_data_by_lang(data)
    st.markdown('''**Start analyzing data...**''')
    analyze_result = {}
    
    # Iterate through each language and each version of data for analysis
    for lang in raw:
        analyze_result[lang]={}
        for version in raw[lang]:
            docs = raw[lang][version]
            analyze_result[lang][version]={}
            st.caption(f'''Start analyzing dataset language {lang}, version {version}, total {len(docs)} batches''')
            
            # Analyze data in batches
            chunk_result= []
            for i, doc in enumerate(docs, start=1):
                st.caption(f'''- Analyzing batch {i}, {len(doc.page_content.split('\n'))} items...''')
                chunk_result.append(_analyze_review_by_lang(doc.page_content, _bedrock_chat))
            st.success(f"Analysis of dataset language {lang}, version {version} completed",icon="✅")
            
            # Merge analysis results
            st.caption(f'''Start merging dataset language {lang}, version {version}''')
            
            analyze_result[lang][version]["xmldata"]=_merge_review_by_lang(''.join(chunk_result), _bedrock_chat)
            st.success(f"Merging dataset language {lang}, version {version} completed",icon="✅")
            
            # Generate report
            st.caption(f'''Start translating and writing report: language {lang}, version {version}''')
            
            st.divider()
            analyze_result[lang][version]["report"] = _write_analysis_report(analyze_result[lang][version]["xmldata"], _bedrock_chat)
            st.success(f'''Report completed: language {lang}, version {version}''',icon="✅")
            st.markdown(analyze_result[lang][version]["report"])
            st.divider()
    return analyze_result
# Compare target version with baseline versions (classified by language)


def analyze_data_by_lang_without_version(data, _bedrock_chat):
    """
    Analyzes review data by language without version information using a language model provided by Amazon Bedrock.



    Example:
        Input:
            data = pd.DataFrame({
                'App Version Code': ['1.0', '2.0', '1.0'],
                'Reviewer Language': ['en', 'fr', 'en'],
                'Review Text': ['Great app', 'Needs improvement', 'Love it']
            })
            _bedrock_chat = some_bedrock_chat_function

        Output:
            {
                'en': {
                    'xmldata': '<version="1.0">...</version>',
                    'report': '**Analysis Report for Version 1.0, Language: English**\n\n...'
                }
                'fr': {
                    'xmldata': '<version="1.0">...</version>',
                    'report': '**Analysis Report for Version 1.0, Language: French**\n\n...'
                },
                ...
            }
        
    """
    data_removed_version = data.drop(columns=['App Version Code'])
    raw = _init_data_by_lang_without_version(data_removed_version)
    st.markdown('''**Start analyzing data...**''')
    analyze_result = {}
    
    for lang in raw:
        docs = raw[lang]
        analyze_result[lang] = {}
        st.caption(f'''Start analyzing dataset language {lang}, total {len(docs)} batches''')

        chunk_result = []
        for i, doc in enumerate(docs, start=1):
            st.caption(f'''- Analyzing batch {i}, {len(doc.page_content.split('\n'))} items...''')
            chunk_result.append(_analyze_review_by_lang_without_version(doc.page_content, _bedrock_chat))
        st.success(f"Analysis of dataset language {lang} completed", icon="✅")

        analyze_result[lang]["xmldata"] = _merge_review_without_version_by_lang(''.join(chunk_result), _bedrock_chat)
        st.success(f"Merging dataset language {lang} completed", icon="✅")

def compare_target_data_by_lang(target_version_no, analyze_result, bedrock_chat):
    """
    Compares the target version's review data with baseline versions for each language.

    Args:
        target_version_no (str): The version number of the target data to compare.
        analyze_result (dict): A nested dictionary containing analysis results for each language and version.
            Structure: {language: {version: {'xmldata': str, 'report': str}}}
        bedrock_chat (function): A function that interfaces with the Amazon Bedrock language model.

    Returns:
        dict: A dictionary where keys are language codes and values are comparison reports in markdown format.

    Example input:
        target_version_no = '2.0'
        analyze_result = {
            'en': {
                '1.0': {'xmldata': '<version="1.0">...</version>', 'report': '...'},
                '2.0': {'xmldata': '<version="2.0">...</version>', 'report': '...'}
            },
            'fr': {
                '1.0': {'xmldata': '<version="1.0">...</version>', 'report': '...'},
                '2.0': {'xmldata': '<version="2.0">...</version>', 'report': '...'}
            }
        }
        bedrock_chat = some_bedrock_chat_function

    Example output:
        {
            'en': '**Comparative Report for Version 2.0, Language Code en**\n\nNew problems...',
            'fr': '**Comparative Report for Version 2.0, Language Code fr**\n\nNouvelles problèmes...'
        }
    """
    compare_result = {}
    for lang, versions in analyze_result.items():
        target_data = ''
        baseline_list = []
        for version, data in versions.items():
            if version == target_version_no:
                target_data = data['xmldata']
            else:
                baseline_list.append(data['xmldata'])
        st.caption(f'''Start comparing: language {lang}, target version {target_version_no}''')
        compare_result[lang] = _compare_analysis_result_by_lang(target_data, ''.join(baseline_list), target_version_no, lang, bedrock_chat)
        st.success(f'''Comparison completed: language {lang}, target version {target_version_no}''', icon="✅")
        st.markdown(compare_result[lang])
    
    return compare_result

# Compare target version with baseline versions (not classified by language)
def compare_target_data(target_version_no, analyze_result, bedrock_chat):
    """
    Compares the target version's review data with baseline versions.

    Args:
        target_version_no (str): The version number of the target data to compare.
        analyze_result (dict): A dictionary containing analysis results for each version.
            Structure: {version: {'xmldata': str, 'report': str}}
        bedrock_chat (function): A function that interfaces with the Amazon Bedrock language model.

    Returns:
        str: A comparison report in markdown format.

    Example input:
        target_version_no = '2.0'
        analyze_result = {
            '1.0': {'xmldata': '<version="1.0">...</version>', 'report': '...'},
            '2.0': {'xmldata': '<version="2.0">...</version>', 'report': '...'}
        }
        bedrock_chat = some_bedrock_chat_function

    Example output:
        '**Comparative Report for Version 2.0**\n\nNew problems...'
    """
    compare_result = {}
    target_data = ''
    baseline_list = []
    for version, data in analyze_result.items():
        if version == target_version_no:
            target_data = data['xmldata']
        else:
            baseline_list.append(data['xmldata'])
    st.caption(f'''Start comparing: target version {target_version_no}''')
    compare_result = _compare_analysis_result(target_data, ''.join(baseline_list), target_version_no, bedrock_chat)
    st.success(f'''Comparison completed: target version {target_version_no}''', icon="✅")
    st.markdown(compare_result)
    return compare_result