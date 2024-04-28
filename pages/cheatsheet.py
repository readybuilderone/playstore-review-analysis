import streamlit as st
import numpy as np
import pandas as pd

st.divider()
st.write("Most objects") # df, err, func, keras!
st.write(["st", "is <", 3]) # see *
# st.write_stream(my_generator)
# st.write_stream(my_llm_stream)

st.text("Fixed width text")
st.markdown("_Markdown_") # see *
st.latex(r""" e^{i\pi} + 1 = 0 """)
st.title("My title")
st.header("My header")
st.subheader("My sub")
st.code("for i in range(8): foo()")


st.divider()
# st.dataframe(my_dataframe)
# st.table(data.iloc[0:10])
st.json({"foo":"bar","fu":"ba"})
st.metric("My metric", 42, 2)

st.divider()
st.radio("Select one:", [1, 2])
    
st.divider()
tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
tab1.write("this is tab 1")
tab2.write("this is tab 2")

st.divider()
with st.chat_message("user"):
    st.write("Hello 👋")
    st.line_chart(np.random.randn(30, 3))
st.chat_input("Say something", key="input1")


with st.container():
    st.chat_input("Say something")