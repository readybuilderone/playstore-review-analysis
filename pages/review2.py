import os 
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from utils import bedrock_wrapper
from utils import review_analyzer

REGION = 'us-west-2'

def _init_session_state():
    # Store the original data from uploaded CSV files
    if 'rawdata' not in st.session_state:
        st.session_state.rawdata = None
    # Store the processed full dataset derived from rawdata
    if 'reviewdata' not in st.session_state:
        st.session_state.reviewdata= None
        
    # store version compare info
    if 'target_version' not in st.session_state:
        st.session_state.target_version= None
    if 'analyze_result' not in st.session_state:
        st.session_state.analyze_result={}
    if 'compare_result' not in st.session_state:
        st.session_state.compare_result=''
    if 'analyze_result_by_lang' not in st.session_state:
        st.session_state.analyze_result_by_lang={}
    if 'compare_result_by_lang' not in st.session_state:
        st.session_state.compare_result_by_lang={}

def _show_review_data_statics(data):
    st.info(f"数据集信息: {len(data)}行", icon="ℹ️")
    
    with st.expander("查看详细", expanded=True, icon="🔎"):
        st.markdown(f'**数据集共包含语言种类:** {len(data['Reviewer Language'].value_counts())}')
        grouped_review_number_by_language = data.groupby('Reviewer Language')['Star Rating'].count().reset_index(name='total review')
        st.bar_chart(grouped_review_number_by_language.set_index('Reviewer Language'))
        
        st.divider()
        st.markdown('**按日期评论数**')
        daily_review_counts = data.groupby('Review Date')['Review Text'].count().reset_index(name='review number')
        st.bar_chart(daily_review_counts.set_index('Review Date'))

        st.divider()
        st.markdown('**按版本评论总数**')
        grouped_review_number_by_version = data.groupby('App Version Code')['Star Rating'].count().reset_index(name='total review')
        st.bar_chart(grouped_review_number_by_version.set_index('App Version Code'))
    
def _show_data_by_rating(data):
    st.write(f"评分分布统计:")
    rating_grouped = data.groupby('Star Rating').size().reset_index(name='RatingCount')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(rating_grouped['RatingCount'], labels=rating_grouped['Star Rating'], autopct='%1.1f%%', pctdistance=0.85)
    ax.axis('equal')  # 确保饼图是圆形的
    st.pyplot(fig)
    
def _show_data_by_version(data):
    st.write(f"版本分布统计:")
    version_grouped = data.groupby('App Version Code').size().reset_index(name='VersionCount')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(version_grouped['VersionCount'], labels=version_grouped['App Version Code'], autopct='%1.1f%%', pctdistance=0.85)
    ax.axis('equal')  # 确保饼图是圆形的
    st.pyplot(fig)

def _analyze_reviews_by_version():
    version_analyze_target_df = st.session_state.reviewdata
    
    all_version = st.session_state.reviewdata['App Version Code'].value_counts()
    col_target, col_baseline = st.columns(2)
    with col_target:
        target_version = st.selectbox('目标版本', all_version.index)
        st.session_state.target_version = target_version
    with col_baseline:
        baseline_version = st.multiselect('基准版本', all_version.index.drop(target_version), [])
        baseline_version = [str(x) for x in baseline_version]
        analyze_version = baseline_version + [str(target_version)]
        version_mask = st.session_state.reviewdata['App Version Code'].isin(analyze_version)
        version_analyze_target_df = st.session_state.reviewdata[version_mask]
    
    _show_data_by_rating(version_analyze_target_df)
    
    analyze_rating = st.multiselect('筛选需要分析的评分', [1, 2, 3, 4, 5], [1,2])
    rating_mask = st.session_state.reviewdata['Star Rating'].isin(analyze_rating)
    version_analyze_target_df = version_analyze_target_df[rating_mask]
    
    st.write('待分析数据: ', len(version_analyze_target_df))
    
    if st.button("点击这个按钮，使用LLM分析评论", type="primary", use_container_width=True):
        with st.status("分析评论...", expanded=True):
            sonnet_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            region_name = REGION
            bedrock_chat = bedrock_wrapper.init_bedrock_chat(model_id=sonnet_id, region_name=region_name)
            st.success("初始化 Bedrock", icon="✅")
            
            st.session_state.analyze_result = review_analyzer.analyze_data(version_analyze_target_df, bedrock_chat)
            st.session_state.compare_result = review_analyzer.compare_target_data(st.session_state.target_version, st.session_state.analyze_result, bedrock_chat)
    
    with st.container(border=True):
        if st.session_state.analyze_result != {}:
            for version, data in st.session_state.analyze_result.items():
                st.success(f'''分析报告：版本{version}''', icon="✅")
                st.markdown(f'''{data['report']}''')
                
        if st.session_state.compare_result != '':
            st.warning(f'''对比报告：目标版本{st.session_state.target_version}, 基准版本{','.join(baseline_version)}''', icon="✅")
            st.markdown(f'''{st.session_state.compare_result}''')
    
    st.divider()
    st.info('按语言筛选数据分析', icon="ℹ️")
    all_lang = st.session_state.reviewdata['Reviewer Language'].value_counts()
    target_lang = st.multiselect('目标语种', all_lang.index, [])
    target_lang = [str(x) for x in target_lang]
    lang_mask = st.session_state.reviewdata['Reviewer Language'].isin(target_lang)
    
    lang_version_analyze_target_df = version_analyze_target_df[lang_mask]
    st.write('待分析数据: ', len(lang_version_analyze_target_df))
    
    st.divider()
    if st.button("点击这个按钮，使用LLM分析目标语言评论", type="primary", use_container_width=True):
        with st.status("分析目标语言评论...", expanded=True):
            st.success("初始化 Bedrock", icon="✅")
            sonnet_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            region_name = REGION
            bedrock_chat = bedrock_wrapper.init_bedrock_chat(model_id=sonnet_id, region_name=region_name)
            st.session_state.analyze_result_by_lang = review_analyzer.analyze_data_by_lang(lang_version_analyze_target_df, bedrock_chat)
            st.session_state.compare_result_by_lang = review_analyzer.compare_target_data_by_lang(st.session_state.target_version, st.session_state.analyze_result_by_lang, bedrock_chat)
    
    with st.container(border=True):
        if st.session_state.analyze_result_by_lang != {}:
            for lang, version in st.session_state.analyze_result_by_lang.items():
                for version, data in st.session_state.analyze_result_by_lang[lang].items():
                    st.success(f'''分析报告: 语言{lang}, 版本{version}''', icon="✅")
                    st.markdown(f'''{data['report']}''')
                
        if st.session_state.compare_result_by_lang != {}:
            for lang, report in st.session_state.compare_result_by_lang.items():
                st.warning(f'''对比报告: 语言{lang}, 目标版本{st.session_state.target_version}, 基准版本{','.join(baseline_version)}''', icon="✅")
                st.markdown(f'''{report}''')

def _analyze_reviews_by_time(data):
    st.write('hello')
    # Get min and max review dates
    min_date = data['Review Date'].min().date()
    max_date = data['Review Date'].max().date()
    
    # Create a date range slider
    date_range = st.slider(
        "选择评论时间范围",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD"
    )
    
    # Filter data based on selected date range
    date_filtered_data = data[(data['Review Date'].dt.date >= date_range[0]) & (data['Review Date'].dt.date <= date_range[1])]
    st.write('选择数据量: ', len(date_filtered_data))
    
    _show_data_by_rating(date_filtered_data)
    
    analyze_rating = st.multiselect('筛选需要分析的评分', [1, 2, 3, 4, 5], [1,2], key='date_analyze_rating')
    rating_mask = date_filtered_data['Star Rating'].isin(analyze_rating)
    date_rating_filtered_data = date_filtered_data[rating_mask]
    
    st.write('待分析数据量: ', len(date_rating_filtered_data))
    
    st.divider()
    _show_data_by_version(date_rating_filtered_data)
    anlyze_version = st.multiselect('筛选需要分析的版本', date_rating_filtered_data['App Version Code'].unique(), key='date_analyze_version')
    version_mask = date_rating_filtered_data['App Version Code'].isin(anlyze_version)
    date_rating_version_filtered_data = date_rating_filtered_data[version_mask]
    
    st.write('待分析数据量: ', len(date_rating_version_filtered_data))
    
    # TODO: Add your analysis logic here using the filtered_data
    if st.button("点击这个按钮，使用LLM分析评论", type="primary", use_container_width=True, key='date_analyze_button'):
        with st.status("分析评论...", expanded=True):
            sonnet_id = "anthropic.claude-3-sonnet-20240229-v1:0"
            region_name = REGION
            bedrock_chat = bedrock_wrapper.init_bedrock_chat(model_id=sonnet_id, region_name=region_name)
            st.success("初始化 Bedrock", icon="✅")
            
            st.session_state.analyze_result_by_time = review_analyzer.analyze_data(date_rating_version_filtered_data, bedrock_chat)
    

st.header("Google Play 应用商店评论分析")
_init_session_state()

uploaded_file_list = st.file_uploader("上传一个或多个文件", accept_multiple_files=True)
if len(uploaded_file_list)>0:
    # init st.session_state.rawdata
    dfs = []
    for csv_file in uploaded_file_list:
        df = pd.read_csv(csv_file, encoding='utf-16')
        dfs.append(df)
    st.session_state.rawdata = pd.concat(dfs, ignore_index=True)
    st.session_state.rawdata.drop_duplicates(keep='first', inplace=True)
    
    # show raw data if user want to
    st.divider()
    if st.checkbox('Show raw data'):
        st.write(st.session_state.rawdata)
        
    # init st.session_state.reviewdata
    st.session_state.reviewdata = st.session_state.rawdata
    st.session_state.reviewdata['App Version Code']= st.session_state.reviewdata['App Version Code'].astype(str)
    st.session_state.reviewdata['App Version Code'] = st.session_state.reviewdata['App Version Code'].fillna('N/A')
    st.session_state.reviewdata['Review Last Update Date and Time'] = pd.to_datetime(st.session_state.reviewdata['Review Last Update Date and Time'], format='mixed')
    st.session_state.reviewdata['Review Date'] = pd.to_datetime(st.session_state.reviewdata['Review Last Update Date and Time'].dt.date)
    st.session_state.reviewdata=st.session_state.reviewdata[['App Version Code', 'Reviewer Language', 'Device', 'Review Date', 
                                'Star Rating', 'Review Title','Review Text']]
    _show_review_data_statics(st.session_state.reviewdata)
    
    
    st.divider()
    tab_time_analyze, tab_version_analyze = st.tabs(["⏳️ 按评论时间分析", "📈 按版本分析"])
    with tab_time_analyze:
        time_analyze_target_df = st.session_state.reviewdata
        _analyze_reviews_by_time(time_analyze_target_df)
    with tab_version_analyze:
        _analyze_reviews_by_version()