import streamlit as st
import time
import pandas as pd
import random
st.title('Hybrid Machine learning Project')
st.header('Select Dataset to predict value!!')
st.subheader('Project Summary:')

summary = '''
This project predicts outcomes using three datasets—Iris, Wine, and Tennis Play. Users can select features to generate predictions for classification tasks:

Iris: Classifies iris plant species based on flower measurements.

Wine: Predicts wine quality based on chemical properties.

Tennis Play: Predicts if a tennis match will be played based on weather conditions.

The system uses machine learning models to provide insights based on the selected features.'''

st.write(summary)


st.sidebar.title('Select Project 🎯 ')
user_project_selection = st.sidebar.radio('Project List: ',['Iris','Wine','Play Tennis'])
# st.sidebar.write(user_project_selection)
# st.write(time.asctime())


temp_df = pd.read_csv(user_project_selection.lower().replace('play ',''))
st.write(temp_df.sample(2))


for i in temp_df.iloc[:,:-1]:
    min_f, max_f = temp_df[i].agg(['min','max']).values
    if str(temp_df[i].agg(['min','max']).dtype) == 'object':
        min_f, max_f = (0,1)  # dropdown pass
    else:
        min_f, max_f = int(min_f), int(max_f)
        if min_f == max_f:
            pass  # pending
        else:
            st.sidebar.slider(f'{i}',min_f,max_f,int(temp_df[i].sample(1).values[0]))











