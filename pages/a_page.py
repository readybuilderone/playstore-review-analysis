import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.write('Use Magic')

df=pd.DataFrame({
    'first column':[1,2,3,4],
    'second column':[5,6,7,8]
})

df

st.write('explicit st write')
st.write(df)

st.write('st.table')
st.table(df)

st.write('st.dataframe')
st.dataframe(df)

st.write('np.rand')
df2=np.random.randn(10,20)
st.dataframe(df2)

dataframe = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))
st.dataframe(dataframe.style.highlight_max(axis=0))

chart_data = pd.DataFrame(
    np.random.randn(20,3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)

import streamlit as st
from datetime import datetime, timedelta
 
# Create a double-ended datetime slider
start_date = datetime(2020, 1, 1)
end_date = start_date + timedelta(days=1000)
 
selected_date_range = st.slider(
    "Select a date range",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, start_date + timedelta(days=7)),
    step=timedelta(days=1),
)