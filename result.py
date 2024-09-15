import pandas as pd
import numpy as np
from Banglanlpdeeplearn.model import model_train
from Banglanlpdeeplearn.text_process import preprocess_text
from Banglanlpdeeplearn.predict import predict_sentiment
df = pd.read_csv('E:/NLP/bangla_sentiment_data.csv')

print(df.head())

df['processed_text'] = df['text'].apply(preprocess_text)

print(df.head())
model1,model2,tokenizer,encoder,X_test,y_test,max_length= model_train(df, 'processed_text', 'label')
loss, accuracy = model2.evaluate(X_test, y_test)
print(f"Model Accuracy: {accuracy}")
new_text ="ভদ্রতা বাড়িতে শিখবেন ইউটিউবে না !!! Okk!!!"
predicted_label = predict_sentiment(new_text, model2, tokenizer, encoder,max_length)
print(predicted_label)
