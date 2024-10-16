import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Read the data
df = pd.read_csv('stock_senti_analysis.csv', encoding='ISO-8859-1')

# Visualize stock sentiment distribution
plt.figure(figsize=(8,8))
sns.countplot(x='Label', data=df)
plt.xlabel('Stock Sentiments (0-Down/Same, 1-Up)')
plt.ylabel('Count')
plt.show()

# Check for missing values and drop if any
print(f"Shape before dropping NA: {df.shape}")
df.dropna(inplace=True)
print(f"Shape after dropping NA: {df.shape}")

# Make a copy of the dataframe
df_copy = df.copy()
df_copy.reset_index(drop=True, inplace=True)

# Split into training and test sets
train = df_copy[df_copy['Date'] < '20150101']
test = df_copy[df_copy['Date'] > '20141231']
print('Train size: {}, Test size: {}'.format(train.shape, test.shape))

# Extracting labels
y_train = train['Label']
y_test = test['Label']

# Selecting columns related to headlines
train = train.iloc[:, 3:28]
test = test.iloc[:, 3:28]

# Download stopwords
nltk.download('stopwords')

# Clean text by removing non-letter characters
train.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)
test.replace(to_replace='[^a-zA-Z]', value=' ', regex=True, inplace=True)

# Rename columns
new_columns = [str(i) for i in range(0, 24)]
print("my columns friend")
train.columns = new_columns
test.columns = new_columns

# Convert to lowercase
for col in new_columns:
    train[col] = train[col].str.lower()
    test[col] = test[col].str.lower()

# Combine the headlines into single strings
train_headlines = [' '.join(str(x) for x in train.iloc[row, 0:25]) for row in range(train.shape[0])]
test_headlines = [' '.join(str(x) for x in test.iloc[row, 0:25]) for row in range(test.shape[0])]
print(train_headlines[0])
print(test_headlines[0])

# Stemming and stopword removal
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_corpus(headlines):
    corpus = []
    for headline in headlines:
        words = headline.split()
        words = [word for word in words if word not in stop_words]
        words = [ps.stem(word) for word in words]
        corpus.append(' '.join(words))
    return corpus

# Preprocess the corpus
train_corpus = preprocess_corpus(train_headlines)
test_corpus = preprocess_corpus(test_headlines)

# Create WordCloud for "down" and "up" words
#down_words = []
#for i in list(y_train[y_train==0].index):
#  down_words.append(train_corpus[i])

#up_words = []
#for i in list(y_train[y_train==1].index):
#  up_words.append(train_corpus[i])


# Plot WordCloud for down_words
#wordcloud1 = WordCloud(background_color='white', width=3000, height=2500).generate(down_words[1])
#plt.figure(figsize=(8,8))
#plt.imshow(wordcloud1)
#plt.axis('off')
#plt.title("Words which indicate a fall in DJIA ")
#plt.show()

#Plot WordCloud for up_words
#wordcloud2 = WordCloud(background_color='white', width=3000, height=2500).generate(up_words[5])
#plt.figure(figsize=(8,8))
#plt.imshow(wordcloud2)
#plt.axis('off')
#plt.title("Words which indicate a rise in DJIA ")
#plt.show()

# Bag of Words model
cv = CountVectorizer(max_features=10000, ngram_range=(2, 2))
X_train = cv.fit_transform(train_corpus).toarray()
X_test = cv.transform(test_corpus).toarray()

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
