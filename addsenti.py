import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

file_path = "stock_senti_analysis.csv"  
df = pd.read_csv(file_path)

sentiment_scores = []

for index, row in df.iterrows():
    combined_text = ' '.join(str(row[col]) for col in df.columns[2:])

    if combined_text != "nan" and combined_text != "":
        sentiment = analyzer.polarity_scores(combined_text)["compound"]
    else:
        sentiment = None 

    sentiment_scores.append(sentiment)

df['Sentiment'] = sentiment_scores

columns = list(df.columns)
columns.insert(2, columns.pop(columns.index('Sentiment')))

df = df[columns]

output_file = "please.csv"
df.to_csv(output_file, index=False)

print(f"Sentiment scores added as 'Sentiment' column and saved to {output_file}")
