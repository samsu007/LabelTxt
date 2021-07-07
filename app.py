import base64
import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")

# Sentiment analysis pipeline
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

st.subheader("Auto Labelling tool for Sentiment Classification Dataset using Huggingface Transformers")

df = pd.read_csv("sample.csv")

predict = classifier(df["text"].values.tolist())

value = []

for i in predict:
    value.append(i['label'])

df['Label'] = value

st.write(df.head())

df.to_csv(index=False)


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="download.csv">Download csv file</a>'
    return href


st.markdown(get_table_download_link(df), unsafe_allow_html=True)
