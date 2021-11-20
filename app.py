import streamlit as st 
import pandas as pd
import spacy
import en_core_web_md
import textacy
import textacy.extract
from textacy.extract import keyterms as kt

def process_data(raw_text):
	return classify_text(raw_text) 


def get_spacy_chunky_text(original_text):    
    doc = nlp_lg(original_text)    
    chunk_text = ''
    
    for chunk in doc.noun_chunks:
        if len(chunk.root.text.lower().strip()) > 3:    
            chunk_text += chunk.root.text.lower().strip() + ' '        

    return nlp_lg(chunk_text)



def get_key_terms(text):
    doc = nlp_lg(text)
    result_rank = kt.textrank(doc, normalize="lemma", topn=5)
    df_doc = pd.DataFrame()
    for i, topic in enumerate(result_rank):    
        df_doc.loc[i, 'Key terms'] = result_rank[i][0]
        df_doc.loc[i, 'Score'] = result_rank[i][1]

    return df_doc


def classify_text(text):
    doc_chunk = get_spacy_chunky_text(text)
    df_doc_topics = pd.DataFrame()

    for i, topic in enumerate(topics_list):
        if (topic.vector_norm):
            df_doc_topics.loc[i, 'Topic'] = topic.text
            df_doc_topics.loc[i, 'Similarity_Score'] = doc_chunk.similarity(topic)
        else:
            print('Topic item without vector: ', topic)

    return df_doc_topics.sort_values(by='Similarity_Score', ascending=False).head(5)

#nlp_lg = spacy.load("en_core_web_md")
nlp_lg = en_core_web_md.load()

df_topics = pd.DataFrame()
df_topics = pd.read_csv('data/topics.csv')
topics_list = [nlp_lg(x) for x in df_topics['Topics']]

st.title("Text classifier")    
raw_text = st.text_area("Paste any text here and see the main topics related")


if st.button('Execute'):
    query_results = process_data(raw_text)
    key_terms_result = get_key_terms(raw_text)            
    with st.expander("Results", expanded=True):
        st.write(query_results)
        st.write(key_terms_result)

