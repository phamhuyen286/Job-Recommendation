#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install streamlit')


# In[2]:


import requests
import pickle
from datetime import datetime
import streamlit as st


# In[3]:


import numpy as np                # mathematical calculations
import pandas as pd               # manipulation of raw data 
import matplotlib.pyplot as plt   # plotting graphs
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


final_jobs = pd.read_csv('E:/Study/Aegis Capstone/job recom/Combined_Jobs_Final.csv')
final_jobs.head()


# In[5]:


final_jobs.shape


# In[6]:


# Listing out all the columns that are present in the data set.

list(final_jobs) 


# In[7]:


print(final_jobs.shape)
final_jobs.isnull().sum()


#  From the above list we see that there are lot of NaN values, perform data cleansing for each and every column

# ## Concatenating the columns ( job corups)
# 

# In[8]:


#subsetting only needed columns and not considering the columns that are not necessary
cols = list(['Job.ID']+['Slug']+['Title']+['Position']+ ['Company']+['City']+['Employment.Type']+['Education.Required']+['Job.Description'])
final_jobs =final_jobs[cols]
final_jobs.columns = ['Job.ID','Slug', 'Title', 'Position', 'Company','City', 'Empl_type','Edu_req','Job_Description']
final_jobs.head() 


# In[9]:


# checking for the null values again.
final_jobs.isnull().sum()


# In[10]:


final_jobs.shape


# In[11]:


#selecting NaN rows of city
nan_city = final_jobs[pd.isnull(final_jobs['City'])]
print(nan_city.shape)
nan_city.head()


# In[12]:


nan_city.groupby(['Company'])['City'].count() 


# We see that there are only 9 companies cities that are having NaN values so I manually adding their head quarters with the help of google search
# 

# In[13]:


#replacing nan with thier headquarters location
final_jobs['Company'] = final_jobs['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')

final_jobs.loc[final_jobs.Company == 'CHI Payment Systems', 'City'] = 'Illinois'
final_jobs.loc[final_jobs.Company == 'Academic Year In America', 'City'] = 'Stamford'
final_jobs.loc[final_jobs.Company == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'
final_jobs.loc[final_jobs.Company == 'Driveline Retail', 'City'] = 'Coppell'
final_jobs.loc[final_jobs.Company == 'Educational Testing Services', 'City'] = 'New Jersey'
final_jobs.loc[final_jobs.Company == 'Genesis Health System', 'City'] = 'Davennport'
final_jobs.loc[final_jobs.Company == 'Home Instead Senior Care', 'City'] = 'Nebraska'
final_jobs.loc[final_jobs.Company == 'St. Francis Hospital', 'City'] = 'New York'
final_jobs.loc[final_jobs.Company == 'Volvo Group', 'City'] = 'Washington'
final_jobs.loc[final_jobs.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'


# In[14]:


final_jobs.isnull().sum()


# In[15]:


#The employement type NA are from Uber so I assume as part-time and full time
nan_emp_type = final_jobs[pd.isnull(final_jobs['Empl_type'])]
print(nan_emp_type)


# In[16]:


#replacing na values with part time/full time
final_jobs['Empl_type']=final_jobs['Empl_type'].fillna('Full-Time/Part-Time')
final_jobs.groupby(['Empl_type'])['Company'].count()
list(final_jobs)


# #   Corpus 

# #### Combining the columns of position,company,city,emp_type and position

# In[17]:


final_jobs["pos_com_city_empType_jobDesc"] = final_jobs["Position"].map(str) + " " + final_jobs["Company"] +" "+ final_jobs["City"]+ " "+final_jobs['Empl_type']+" "+final_jobs['Job_Description']
final_jobs.pos_com_city_empType_jobDesc.head()


# In[18]:


#removing unnecessary characters between words separated by space between each word of all columns to make the data efficient
final_jobs['pos_com_city_empType_jobDesc'] = final_jobs['pos_com_city_empType_jobDesc'].str.replace('[^a-zA-Z \n\.]'," ") #removing unnecessary characters
final_jobs.pos_com_city_empType_jobDesc.head()


# In[19]:


#converting all the characeters to lower case
final_jobs['pos_com_city_empType_jobDesc'] = final_jobs['pos_com_city_empType_jobDesc'].str.lower() 
final_jobs.pos_com_city_empType_jobDesc.head()


# In[20]:


final_all = final_jobs[['Job.ID', 'pos_com_city_empType_jobDesc']]
# renaming the column name as it seemed a bit complicated
final_all = final_jobs[['Job.ID', 'pos_com_city_empType_jobDesc']]
final_all = final_all.fillna(" ")

final_all.head()


# In[21]:


print(final_all.head(1))


# In[22]:


np.save('E:/Study/Aegis Capstone/job recom/job_final.npy', final_all)


#   ##  NLTK Process
# 

# In[23]:


# Setup
get_ipython().system('pip install -q wordcloud')
import wordcloud

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 

import pandas as pd
import matplotlib.pyplot as plt
import io
import unicodedata
import numpy as np
import re
import string


# In[24]:


pos_com_city_empType_jobDesc = final_all['pos_com_city_empType_jobDesc']
#removing stopwords and applying potter stemming
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer =  PorterStemmer()
stop = stopwords.words('english')
only_text = pos_com_city_empType_jobDesc.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
only_text.head()


# Splitting each word in a row separated by space.

# In[25]:


only_text = only_text.apply(lambda x : filter(None,x.split(" ")))
print(only_text.head())


# In[26]:


only_text = only_text.apply(lambda x : [stemmer.stem(y) for y in x])
print(only_text.head())


# In the above code, we separated each letter in a word separated by comma, now, in this step, we join the words(x)
# 

# In[27]:


only_text = only_text.apply(lambda x : " ".join(x))
print(only_text.head())


# In[28]:


#adding the featured column back to pandas
final_all['text']= only_text
# As we have added a new column by performing all the operations using lambda function, we are removing the unnecessary column
#final_all = final_all.drop("pos_com_city_empType_jobDesc", 1)

list(final_all)
final_all.head()


# In[29]:


# in order to save this file for a backup
#final_all.to_csv("job_data.csv", index=True)


# # TF-IDF ( Term Frequency - Inverse Document Frequency ) 
# 

# In[30]:



#initializing tfidf vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

tfidf_vectorizer = TfidfVectorizer()

tfidf_jobid = tfidf_vectorizer.fit_transform((final_all['text'])) #fitting and transforming the vector
tfidf_jobid


# # User query Corpus
# Take another dataset called job views.

# In[31]:


#Consider a  new data set and  taking the datasets job view, position of interest, experience of the applicant into consideration for creating a query who applied for job
job_view = pd.read_csv(r'E:\Study\Aegis Capstone\job recom\Job_Views.csv')
job_view.head()


# In[32]:


#subsetting only needed columns and not considering the columns that are not necessary as we did that earlier.
job_view = job_view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]

job_view["pos_com_city"] = job_view["Position"].map(str) + "  " + job_view["Company"] +"  "+ job_view["City"]

job_view['pos_com_city'] = job_view['pos_com_city'].str.replace('[^a-zA-Z \n\.]',"")

job_view['pos_com_city'] = job_view['pos_com_city'].str.lower()

job_view = job_view[['Applicant.ID','pos_com_city']]

job_view.head()


# In[33]:


np.save('E:/Study/Aegis Capstone/job recom/job_view.npy', job_view)


# ### Experience
# We take experience of all the applicants who applied for the job and we are comaring the point of interest with the jobs that are present in our previous data.

# In[34]:


#Experience
exper_applicant = pd.read_csv(r'E:\Study\Aegis Capstone\job recom\Experience.csv')
exper_applicant.head()


# In[35]:


#taking only Position
exper_applicant = exper_applicant[['Applicant.ID','Position.Name']]

#cleaning the text
exper_applicant['Position.Name'] = exper_applicant['Position.Name'].str.replace('[^a-zA-Z \n\.]',"")

exper_applicant.head()
#list(exper_applicant)


# In[36]:


exper_applicant['Position.Name'] = exper_applicant['Position.Name'].str.lower()
exper_applicant.head(10)


# In[37]:


exper_applicant =  exper_applicant.sort_values(by='Applicant.ID')
exper_applicant = exper_applicant.fillna(" ")
exper_applicant.head(20)


# same applicant has 3 applications 100001 in sigle line

# In[38]:


#adding same rows to a single row
exper_applicant = exper_applicant.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()
exper_applicant.head(20)


# In[39]:


np.save('E:/Study/Aegis Capstone/job recom/exper_application.npy', exper_applicant)


# ### Position of Interest

# In[40]:


#Position of interest
poi =  pd.read_csv('E:\Study\Aegis Capstone\job recom\Positions_Of_Interest.csv')
poi = poi.sort_values(by='Applicant.ID')
poi.head()


# In[41]:


# There is no need of application and updation becuase there is no deadline mentioned in the website ( assumption) hence we are droping unimportant attributes
poi = poi.drop('Updated.At', 1)
poi = poi.drop('Created.At', 1)

#cleaning the text
poi['Position.Of.Interest']=poi['Position.Of.Interest'].str.replace('[^a-zA-z \n\.]',"")
poi['Position.Of.Interest']=poi['Position.Of.Interest'].str.lower()
poi = poi.fillna(" ")
poi.head(20)


# In[42]:


poi = poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()
poi.head()


# In[43]:


np.save('E:/Study/Aegis Capstone/job recom/poi.npy', poi)


# ### Merging

# pos_com city, position_name

# In[44]:


#merging jobs and experience dataframes
out_joint_jobs = job_view.merge(exper_applicant, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
print(out_joint_jobs.shape)
out_joint_jobs = out_joint_jobs.fillna(' ')
out_joint_jobs = out_joint_jobs.sort_values(by='Applicant.ID')
out_joint_jobs.head()


# In[45]:


#merging position of interest with existing dataframe
joint_poi_exper_view = out_joint_jobs.merge(poi, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')
joint_poi_exper_view = joint_poi_exper_view.fillna(' ')
joint_poi_exper_view = joint_poi_exper_view.sort_values(by='Applicant.ID')
joint_poi_exper_view.head()


# In[46]:


#combining all the columns

joint_poi_exper_view["pos_com_city1"] = joint_poi_exper_view["pos_com_city"].map(str) + joint_poi_exper_view["Position.Name"] +" "+ joint_poi_exper_view["Position.Of.Interest"]

joint_poi_exper_view.head()


# In[47]:


final_poi_exper_view = joint_poi_exper_view[['Applicant.ID','pos_com_city1']]
final_poi_exper_view.head()


# In[48]:


final_poi_exper_view.columns = ['Applicant_id','pos_com_city1']
final_poi_exper_view.head()


# In[49]:


final_poi_exper_view = final_poi_exper_view.sort_values(by='Applicant_id')
final_poi_exper_view.head()


# In[50]:


final_poi_exper_view['pos_com_city1'] = final_poi_exper_view['pos_com_city1'].str.replace('[^a-zA-Z \n\.]',"")
final_poi_exper_view.head()


# In[51]:


final_poi_exper_view['pos_com_city1'] = final_poi_exper_view['pos_com_city1'].str.lower()
final_poi_exper_view.head()


# In[52]:



final_poi_exper_view = final_poi_exper_view.reset_index(drop=True)
final_poi_exper_view.head()


# select random row of 6999

# In[53]:


#taking a user
u = 6999
index = np.where(final_poi_exper_view['Applicant_id'] == u)[0][0]
user_q = final_poi_exper_view.iloc[[index]]
user_q


# # Using vector space model ( Cosine similarity )
# 

# In[54]:


#creating tf-idf of user query and computing cosine similarity of user with job corpus
from sklearn.metrics.pairwise import cosine_similarity
user_tfidf = tfidf_vectorizer.transform(user_q['pos_com_city1'])
output = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)


# In[55]:


output2 = list(output)


# In[56]:


import sys


# In[57]:


#getting the job id's of the recommendations
top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:50]
recommendation = pd.DataFrame(columns = ['ApplicantID', 'JobID'])
count = 0
for i in top:
  recommendation.at[count, 'ApplicantID']=u
  recommendation.at[count,'JobID'] = final_all['Job.ID'][i]
  count += 1


# cosine similarity score

# In[58]:


recommendation


# In[59]:


#getting the job ids and their data
nearestjobs = recommendation['JobID']
job_description = pd.DataFrame(columns = ['JobID','text'])
for i in nearestjobs:
    index = np.where(final_all['Job.ID'] == i)[0][0]    
    job_description.at[count, 'JobID']=i
    job_description.at[count, 'text']= final_all['text'][index]
    count += 1
    


# In[60]:


#printing the jobs that matched the query
job_description


# In[61]:


job_description.to_csv("recommended_content.csv")


# In[67]:


result = final_all.to_csv("job_data.csv", index=False)


# # Job Recommendation App

# In[63]:


st.title("Job Recommendation")


# In[68]:


st.image("image.png", use_column_width=True) 


# In[73]:


$ streamlit hello


# In[70]:


st.header("")

input_msg = st.text_input("")

st.subheader("Press enter to check the prediction...")

st.write(result)


# In[71]:


def main():
    df = result
    page = st.sidebar.selectbox("Choose a page", ["Homepage", "Exploration"])

    if page == "Homepage":
        st.header("This is your data explorer.")
        st.write("Please select a page on the left.")
        st.write(df)
    elif page == "Exploration":
        st.title("Data Exploration")
        x_axis = st.selectbox("Choose a variable for the x-axis", df.columns, index=3)
        y_axis = st.selectbox("Choose a variable for the y-axis", df.columns, index=4)
        visualize_data(df, x_axis, y_axis)

@st.cache
def load_data():
    df = data.cars()
    return df

def visualize_data(df, x_axis, y_axis):
    graph = alt.Chart(df).mark_circle(size=60).encode(
        x=x_axis,
        y=y_axis,
        color='Origin',
        tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    ).interactive()

    st.write(graph)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




