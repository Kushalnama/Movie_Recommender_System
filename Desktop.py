#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


movie=pd.read_csv("tmdb_5000_movies.csv")
movie.head(10)


# In[3]:


credits=pd.read_csv("tmdb_5000_credits.csv")
credits.head(1)


# In[4]:


# Merge the Datasets on the basis of "title column". 
data = movie.merge(credits,on="title")


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


movies=data[["movie_id","title","overview","genres","keywords","cast","crew"]]
movies.sample(4)


# In[9]:


# Removing null and duplicated values 
movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.isnull().sum()


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


# string list :-  '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
##          list = ["Action","Adventure","Fantasy", "Science Fiction"]


# In[15]:


# we use "ast" to convert the string list into a list.
import ast 
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[16]:


def change(name):
    list=[]
    for i in ast.literal_eval(name):
        list.append(i["name"])
    return list


# In[17]:


movies['genres'].apply(change)


# In[18]:


movies['genres'] = movies['genres'].apply(change) 


# In[19]:


movies.head(1)


# In[20]:


movies.iloc[0].keywords


# In[21]:


movies['keywords'] = movies['keywords'].apply(change)


# In[22]:


movies.head(1)


# In[23]:


movies['cast'][0]


# In[24]:


def change1(name):
    list=[]
    count=0
    for i in ast.literal_eval(name):
        if count != 4: 
            list.append(i["name"])
            count+=1
        else:
            break
    return list


# In[25]:


movies["cast"].apply(change1)


# In[26]:


movies["cast"]=movies["cast"].apply(change1)


# In[27]:


movies.head(2)


# In[28]:


movies["crew"][0]


# In[29]:


def change2(name):
    list=[]
    for i in ast.literal_eval(name):
        if i['job']== "Director":
            list.append(i["name"])
            break
    return list


# In[30]:


movies["crew"].apply(change2)


# In[31]:


movies["crew"] = movies["crew"].apply(change2)


# In[32]:


movies.head()


# In[33]:


# "overview" is in string we wnat to convert it into an string.
movies['overview'][0]


# In[34]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[35]:


movies.head()


# In[36]:


# In this we replace, the space " " to ""

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[37]:


movies.head()


# In[38]:


movies["tags"] = movies['overview'] + movies['genres']+ movies['keywords'] + movies['cast'] + movies['crew']


# In[39]:


movies.head()


# In[40]:


n_movies = movies[["movie_id","title","tags"]]


# In[41]:


n_movies


# In[42]:


# Covert list into string 
n_movies["tags"] = n_movies['tags'].apply(lambda x:(" ".join(x)))


# In[43]:


n_movies.head()


# In[44]:


n_movies["tags"][0]


# In[45]:


n_movies['tags'] = n_movies['tags'].apply(lambda x:x.lower())


# In[46]:


n_movies.head()


# In[47]:


# Stemming :- It convert the text into its root words


# In[48]:


import nltk


# In[49]:


from nltk.stem import PorterStemmer


# In[50]:


ps=PorterStemmer()


# In[51]:


def stem(text):
    y =[]
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[52]:


n_movies['tags'] = n_movies['tags'].apply(stem)


# In[53]:


n_movies.head()


# In[54]:


# Bag of Words :- It count the total no. of similar words and then put it to an dictionary.


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer


# In[56]:


vectorizer = CountVectorizer(max_features=5000,stop_words="english")


# In[57]:


vectors = vectorizer.fit_transform(n_movies['tags']).toarray()


# In[58]:


vectors


# In[59]:


vectors[0]


# In[60]:


## COSINE Distance :- similarity is inversally proportional to distance.


# In[61]:


from sklearn.metrics.pairwise import cosine_similarity


# In[62]:


similar = cosine_similarity(vectors)


# In[63]:


similar[0]


# In[64]:


sorted(list(enumerate(similar[0])),reverse=True,key=lambda x:x[1])[1:11]


# In[65]:


def recommend(movie):
    movie_index = n_movies[n_movies['title']== movie].index[0]
    distance = similar[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:11]
    
    for i in movie_list:
        print(n_movies.iloc[i[0]]['title'])
 


# In[68]:


recommend("Independence Day")


# In[67]:


n_movies.iloc[1216]['title']


# In[ ]:





# In[ ]:





# In[ ]:




