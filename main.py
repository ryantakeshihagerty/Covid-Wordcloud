import datetime
import pickle
from collections import Counter
import pandas as pd
from pathlib import Path
import spacy
from newsapi import NewsApiClient
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_lg')
newsapi = NewsApiClient(api_key='256f373818d54a11b7e3f382b03eade4')

cur_date = datetime.datetime.today().strftime('%Y-%m-%d')
from_date = (datetime.datetime.today() - datetime.timedelta(29)).strftime('%Y-%m-%d')


# Get articles
def get_page(num):
    return newsapi.get_everything(q='coronavirus', language='en', from_param=from_date, to=cur_date,
                                  sort_by='relevancy', page=num)


articles = list(map(get_page, range(1, 6)))

# Save articles using pickle library
filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
filepath = str(Path().absolute()) + filename  # Get path to .pckl file
pickle.dump(loaded_model, open(filepath, 'wb'))

# Clean data
data = []
for i, article in enumerate(articles):
    for x in article['articles']:
        titles = x['title']
        descriptions = x['description']
        content = x['content']
        data.append({'title': titles, 'desc': descriptions, 'content': content})

df = pd.DataFrame(data)
df = df.dropna()
df.head()


def get_keywords(text):
    doc = nlp(text)
    result = []
    pos_tag = ['VERB', 'NOUN', 'PROPN']
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)
    return result


results = []
for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords(content)).most_common(5)])

df['keywords'] = results

df.to_csv('data.csv')

# Create wordcloud
text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
