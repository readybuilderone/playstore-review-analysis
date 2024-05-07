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

if "rawdata" not in st.session_state:
    st.session_state.rawdata = st.empty()
    st.session_state.target_version=st.empty()

st.header("App Review")

def _show_raw_data_statics(data):
    st.divider()
    st.info('数据集信息', icon="ℹ️")
    st.write('数据集原始数据行数: ', len(data))
    
    st.divider()
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
    

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    st.session_state.rawdata = pd.read_csv(uploaded_file, encoding='utf-16')
    st.divider()
    if st.checkbox('Show raw data'):
        st.write(st.session_state.rawdata)
    
    st.session_state.reviewdata = st.session_state.rawdata
    st.session_state.reviewdata['App Version Code']= st.session_state.reviewdata['App Version Code'].astype(str)
    st.session_state.reviewdata['App Version Code'] = st.session_state.reviewdata['App Version Code'].fillna('N/A')
    st.session_state.reviewdata['Review Last Update Date and Time'] = pd.to_datetime(st.session_state.reviewdata['Review Last Update Date and Time'], format='mixed')
    st.session_state.reviewdata['Review Date'] = pd.to_datetime(st.session_state.reviewdata['Review Last Update Date and Time'].dt.date)
    
    _show_raw_data_statics(st.session_state.reviewdata)
    
    st.divider()
    st.info('筛选数据进行分析', icon="ℹ️")
    all_version=st.session_state.reviewdata['App Version Code'].value_counts()
    
    target_version = st.selectbox(
        '目标版本',
        all_version.index
    )
    st.session_state.target_version=target_version
    
    baseline_version = st.multiselect(
        '基准版本',
        all_version.index.drop(target_version),
        [])
    baseline_version = [str(x) for x in baseline_version]
    analyze_version = baseline_version + [str(target_version)]
    version_mask=st.session_state.reviewdata['App Version Code'].isin(analyze_version)
    
    all_lang=st.session_state.reviewdata['Reviewer Language'].value_counts()
    target_lang = st.multiselect(
        '目标语种',
        all_lang.index,
        [])
    target_lang = [str(x) for x in target_lang]
    lang_mask=st.session_state.reviewdata['Reviewer Language'].isin(target_lang)
    
    target_df=st.session_state.reviewdata[version_mask][lang_mask]
    st.write('选中数据行数: ', len(target_df))
    
    # st.write(target_df.head(5))
    
    st.divider()
    if len(target_df) > 0:
        st.markdown('''**评论 版本/语言 平均分：**''')
        
        st.divider()
        pivot_table = pd.pivot_table(
            target_df,
            values='Star Rating',
            index='Reviewer Language',
            columns='App Version Code',
            aggfunc='mean'
        )
        highlight_low_rating = 'background-color: white; color: orange'
        styled_pivot_table = pivot_table.style.applymap(
            lambda x: highlight_low_rating if x < 4 else ''
        )
        st.dataframe(styled_pivot_table)

        st.divider()
        grouped = target_df.groupby(['App Version Code', 'Reviewer Language'])['Star Rating'].mean().reset_index()
        plot_data = grouped.pivot(index='App Version Code', columns='Reviewer Language', values='Star Rating')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = plot_data.plot(kind='bar', ax=ax)
        ax.set_xlabel('App Version Code')
        ax.set_ylabel('Review Rating')
        st.pyplot(fig)
        
        st.divider()
        st.markdown('''**评论 版本/语言 数量：**''')
        st.divider()
        pivot_table = pd.pivot_table(
            target_df,
            values='Review Text',
            index='Reviewer Language',
            columns='App Version Code',
            aggfunc='count'
        )
        st.dataframe(pivot_table)
        grouped = target_df.groupby(['App Version Code', 'Reviewer Language'])['Review Text'].count().reset_index()
        plot_data = grouped.pivot(index='App Version Code', columns='Reviewer Language', values='Review Text')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax = plot_data.plot(kind='bar', ax=ax)
        ax.set_xlabel('App Version Code')
        ax.set_ylabel('Review Text Count')
        st.pyplot(fig)
    
        
        st.divider()
        st.info('使用Bedrock分析目标数据', icon="ℹ️")
        analyze_rating = st.multiselect(
            '需要分析的评分',
            [1, 2, 3, 4, 5],
            [1,2])
        rating_mask=st.session_state.reviewdata['Star Rating'].isin(analyze_rating)
        target_df=target_df[rating_mask]
        st.write(f"待分析数据:{len(target_df)}")
        
        
        if st.button("点击这个按钮，触发LLM分析评论", type="primary", use_container_width=True):
            with st.status("分析评论...", expanded=True):
                target_df=target_df[['App Version Code', 'Reviewer Language', 'Device', 'Review Date', 
                                     'Star Rating', 'Review Title','Review Text']]
                
                st.success("初始化 Bedrock",icon="✅")
                sonnet_id = "anthropic.claude-3-sonnet-20240229-v1:0"
                region_name='us-west-2'
                bedrock_chat=bedrock_wrapper.init_bedrock_chat(model_id=sonnet_id, region_name=region_name)

                analyze_results = review_analyzer.analyze_data(target_df, bedrock_chat)
                compare_results = review_analyzer.compare_target_data(st.session_state.target_version, analyze_results, bedrock_chat)
                # st.json(analyze_results)
                
            # for lang, versions in analyze_results.items():
            #     for version, data in versions.items():
            #         st.divider()
            #         st.write(f"{lang}-{version}")
            #         data=analyze_results[lang][version]['report']
            #         st.markdown(data)
                

                
