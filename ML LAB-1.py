#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
 


# In[3]:


from sklearn.model_selection import train_test_split
data = [20, 4, 12, 9, 0, 10]
labels = ["A", "B", "B", "A", "C", "A"]
train, test = train_test_split(data, test_size=0.5)
print("Splitting into equal parts:")
print("Train Split:", train)
print("Test Split:", test)
train, test = train_test_split(data, test_size=0.2)
print("\nSplitting into different parts:")
print("Train Split:", train)
print("Test Split:", test)
train_data, test_data, train_labels, test_labels = train_test_split(data, labels) 
print("\nSplitting multiple lists:")
print("Train Data:", train_data)
print("Test Data:", test_data)
print("Train Labels:", train_labels)
print("Test Labels:", test_labels)


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
text = ["John is playing basketball", "Mary is playing tennis"]
vectorizer = CountVectorizer()
vectorizer.fit(text)
X = vectorizer.transform(text)
print(X.toarray())


# In[5]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
text = ["John is playing basketball", "Mary is playing tennis", "John is playing tennis"]
vectorizer = CountVectorizer()
vectorizer.fit(text)
X = vectorizer.transform(text)
clf = MultinomialNB()
clf.fit(X, [0, 1, 1])
new_doc = ["Mary is playing basketball"]
new_doc_counts = vectorizer.transform(new_doc)
print(clf.predict(new_doc_counts))


# In[16]:


import pandas as pd
data=pd.read_csv("apple_quality.csv")
df=pd.DataFrame(data)
df=df.drop("A_id",axis=1)
df=df.drop(4000,axis=0)
df


# In[24]:


X=df[['Size','Weight','Sweetness','Crunchiness','Juiciness','Ripeness','Acidity']]
Y=df['Quality']
X


# In[25]:


Y


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test= train_test_split(X,Y, test_size=0.5,random_state=42)
clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[46]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",cm)


# In[48]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plot_tree(clf,filled=True)
plt.show()


# In[ ]:




