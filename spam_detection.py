import pandas as pd
spam = pd.read_csv("spam.csv")
spam.drop(spam.columns[spam.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

#Text Filetring
spec_chars = ["!",'"',"#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]
for char in spec_chars:
    spam['v2'] = spam['v2'].str.replace(char, ' ')
    spam['v2'] = spam['v2'].str.split().str.join(" ")

#Convert column to lowercase
spam["v2"]= spam["v2"].str.lower()

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(spam, test_size=0.2, random_state = 42)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(spam, spam['v1']):
    strat_train_set = spam.loc[train_index]
    strat_test_set = spam.loc[test_index]

features = strat_train_set.drop('v1', axis=1)
labels = strat_train_set['v1'].copy()

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(features).toarray()

tfidfconverter = TfidfTransformer()
spam_strat_tr = tfidfconverter.fit_transform(X).toarray()


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(spam_strat_tr, labels)

