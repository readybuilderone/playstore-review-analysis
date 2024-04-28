from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st

def _split_df_to_docs(df, chunk_size=300000):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    docs = text_splitter.create_documents([df.to_string(index=False)])
    return docs


def _analyze_review(content, bedrock_chat):
    analyze_prompt = PromptTemplate(
        template="""
        \n\nHuman: 

        You are an AI assistant trained to identify and categorize user negative reviews.
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

    insight_chain = analyze_prompt | bedrock_chat | StrOutputParser()
    
    result_list=[]
    for chunk in insight_chain.stream({
        "document": {content},
    }):
        # print(chunk, end="", flush=True)
        result_list.append(chunk)
        
    return ''.join(result_list)

def _merge_review(content, bedrock_chat):
    merge_prompt = PromptTemplate(
    template="""
        You are an AI assistant. 
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

    merge_chain = merge_prompt | bedrock_chat | StrOutputParser()
    result_list=[]
    for chunk in merge_chain.stream({
        "reviews": {content},
    }):
        # print(chunk, end="", flush=True)
        result_list.append(chunk)
        
    return ''.join(result_list)

def _write_analysis(content, bedrock):
    writing_prompt = PromptTemplate(
    template="""
        You're an AI assistant who's proficient in multiple languages and good at writing.
        You'll be provided with a batch of review issues categoried by version in xml format, your task is to analyze the material provided in the <content></content> tag and then write an analysis in markdown format. 
        You need to follow the instructions in <instructions></instructions> tags.

        <content> {reviews} </content>

        <instructions>
        - The analysis must be in markdown format, you can use bold, but you must not use heading.
        - The analysis must be in simple chinese.
        - The analysis must contain three sections: Summary, Key Issues Analysis and Conclusion.
        - You must calculate the total number of counts by adding up the counts of each issue, and you must include this total number of counts in the Summary, outputting as: "The total number of unsatisfactory feedback from players is xx".
        - For each issue, you must analysis it in bullet format, and you must include what the count of the Issue is, what the percentage of the Issue count is to the total of all Issue counts, outputting as: "Count:xxx, Percentage: yy.yy%".
        - For each issue analysis, you must explain in detail why and what the player is dissatisfied with in a new line.
        </instructions>

        \n\nAssistant:
        """,
        input_variables=["reviews"]
    )

    writing_chain = writing_prompt | bedrock | StrOutputParser()
    result_list=[]
    for chunk in writing_chain.stream({
        "reviews": {content},
    }):
        # print(chunk, end="", flush=True)
        # st.write_stream(chunk)
        result_list.append(chunk)
        
    return ''.join(result_list)

def _init_data(data):
    st.markdown('''**开始拆分数据...**''')
    raw={}
    for lang in data['Reviewer Language'].unique():
        raw[lang]={}
        for version in data['App Version Code'].unique():
            # print(f"{lang}, {version}")
            target_data = data[(data['Reviewer Language']== lang) & (data['App Version Code']==version)]
            # print(f"target_data row: {len(target_data)}")
            docs = _split_df_to_docs(target_data)
            raw[lang][version]= docs
            # print(f"target data doc number: {len(docs)}")
            st.success(f"拆分数据:语言{lang}, 版本:{version}, 共{len(docs)} 批",icon="✅")
    return raw

def analyze_data(data, bedrock_chat):
    raw = _init_data(data)
    st.markdown('''**开始分析数据...**''')
    analyze_result = {}
    for lang in raw:
        analyze_result[lang]={}
        for version in raw[lang]:
            docs = raw[lang][version]
            st.caption(f'''开始分析数据集语言{lang}, 版本{version}, 共{len(docs)}批''')
            chunk_result= []
            for i, doc in enumerate(docs, start=1):
                st.caption(f'''- 分析第{i}批数据, {len(doc.page_content.split('\n'))} 条...''')
                chunk_result.append(_analyze_review(doc.page_content, bedrock_chat))
            st.success(f"分析数据集语言{lang}, 版本{version} 完成",icon="✅")
            
            st.caption(f'''开始合并数据集语言{lang}, 版本{version}''')
            analyze_result[lang][version]=_merge_review(''.join(chunk_result), bedrock_chat)
            st.success(f"合并数据集语言{lang}, 版本{version} 完成",icon="✅")
            # todo write article
            st.caption(f'''开始翻译并撰写报告：语言{lang}, 版本{version}''')
            
            st.divider()
            # _write_analysis(analyze_result[lang][version], bedrock_chat)
            analysis_report = _write_analysis(analyze_result[lang][version], bedrock_chat)
            st.success(f'''完成报告：语言{lang}, 版本{version}''',icon="✅")
            st.markdown(analysis_report)
            st.divider()
    return analyze_result