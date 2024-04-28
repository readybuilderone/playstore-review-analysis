import streamlit as st
import pandas as pd
import numpy as np
import time

st.write('B Page')

st.write('Widget')

st.write('--'*80)
x=st.slider('x')
st.write(x, 'result is: ', x*x)

st.write('--'*80)
st.text_input("Your Name", key='name')
st.write('Your name is: ', st.session_state.name)

st.write('--'*80)
if st.checkbox('Show Dataframe:'):
    df=pd.DataFrame(np.random.randn(10,3), columns=['a','b','c'])
    df

st.write('--'*80)
option = st.selectbox(
    'Which number do you like best?',
    [1,10,100])

'You selected: ', option

st.write('--'*80)
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ['Email', 'Home phone', 'Mobile phone']
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)

st.write('--'*80)
left_column, right_column = st.columns(2)
# You can use a column just like st.sidebar:
left_column.button('Press me!')

# Or even better, call Streamlit functions inside a "with" block:
with right_column:
    chosen = st.radio(
        'Sorting hat',
        ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"))
    st.write(f"You are in {chosen} house!")
    

st.write('--'*80)
'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'

st.divider()
st.info('This is a purely informational message', icon="ℹ️")