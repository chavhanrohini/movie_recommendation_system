#!/usr/bin/env python
# coding: utf-8

# In[72]:


import numpy as np 
import pandas as pd


# In[73]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[74]:


movies.head(1)


# In[75]:


credits.head(1)


# In[76]:


movies = movies.merge(credits, on ='title')


# In[77]:


movies.head(1)


# In[78]:


movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]


# In[79]:


movies.head(1)


# In[80]:


movies.isnull().sum()


# In[81]:


movies.dropna(inplace = True)


# In[82]:


movies.isnull().sum()


# In[83]:


movies.duplicated().sum()


# In[84]:


movies.iloc[0].genres


# In[85]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[86]:


import ast

def convert(obj):
    try:
        l= []
        for i in ast.literal_eval(obj):
            l.append(i['name'])
        return l
    except (SyntaxError, ValueError):
        return []


# In[87]:


movies.dropna(inplace=True)


# In[88]:


movies['genres'].apply(convert)


# In[89]:


movies['genres'] = movies['genres'].apply(convert)


# In[90]:


movies.head()


# In[91]:


movies['keywords']=  movies['keywords'].apply(convert)


# In[92]:


movies.head()


# In[93]:


movies['cast'][0]


# In[94]:


import ast

def convert3(obj):
    try:
        l= []
        counter = 0
        for i in ast.literal_eval(obj):
            if counter != 3:
                l.append(i['name'])
                counter += 1
            else :
                break
        return l
    except (SyntaxError, ValueError):
        return []


# In[95]:


movies['cast'].apply(convert3)


# In[96]:


movies['cast'] = movies['cast'].apply(convert3)


# In[97]:


movies.head()


# In[98]:


movies['crew'][0]


# In[99]:


def director(obj):
    try:
        l= []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                l.append(i['name'])
                break
        return l
    except (SyntaxError, ValueError):
        return []


# In[100]:


movies['crew'].apply(director)


# In[101]:


movies['crew']= movies['crew'].apply(director)


# In[102]:


movies.head()


# In[103]:


movies['overview'][0]


# In[104]:


#converting into list
movies['overview'].apply(lambda x:x.split())


# In[105]:


movies['overview']= movies['overview'].apply(lambda x:x.split())


# In[106]:


movies.head()


# In[107]:


#Replacing space with nothing, removing space
movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[108]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])


# In[109]:


movies.head()


# In[110]:


movies['tags'] = movies['overview']+ movies['genres']+ movies['keywords']+ movies['cast']+movies['crew']


# In[111]:


movies.head()


# In[112]:


#creating new dataframe having required column
new_df= movies[['movie_id', 'title','tags']]


# In[113]:


new_df.head()


# In[114]:


#converting into string
new_df['tags'].apply(lambda x: " ".join(x))


# In[115]:


new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# In[116]:


new_df.head()


# In[117]:


new_df['tags'][0]


# In[ ]:





# In[118]:


new_df['tags']= new_df['tags'].apply(lambda x:x.lower())


# In[119]:


new_df.head()


# In[120]:


import nltk


# In[121]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[122]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[123]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[124]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[125]:


pip install scikit-learn


# In[126]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words= 'english')


# In[127]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[128]:


vectors


# In[129]:


vectors[0]


# In[130]:


cv.get_feature_names()


# In[131]:


from sklearn.metrics.pairwise import cosine_similarity


# In[132]:


cosine_similarity(vectors)


# In[133]:


cosine_similarity(vectors).shape


# In[134]:


similarity = cosine_similarity(vectors)


# In[135]:


sorted(list(enumerate(similarity[0])), reverse=True, key= lambda x:x[1])[1:6]


# In[136]:


def reccomend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key= lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
   


# In[137]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[138]:


reccomend('Avatar')


# In[139]:


new_df.iloc[1216].title


# In[140]:


import pickle


# In[141]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl', 'wb'))


# In[142]:


pickle.dump(similarity,open('similarity.pkl', 'wb'))

