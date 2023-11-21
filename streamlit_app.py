import streamlit as st
import os
import pandas as pd
import numpy as np
import re
pd.set_option('display.max_colwidth', None)
# 行数の最大表示数を設定（全ての行を表示）
#pd.set_option('display.max_rows', None)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import streamlit as st
wd = os.getcwd()
data_dir = os.path.join(wd,'drive/MyDrive/datamix/integral_step/dataset')

@st.cache_data
def load_data():
    # ここでDataFrameをロードします
    article_df = pd.read_csv(f'{data_dir}/tokenized_streamlit.csv')
    return article_df

article_df = load_data()

@st.cache_data
def load_data():
    # ここでDataFrameをロードします
    tfidf_matrix = pd.read_csv(f'{data_dir}/tfidf_matrix.csv')
    return tfidf_matrix

tfidf_matrix = load_data()

vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
tfidf_text = vectorizer.fit_transform(article_df.text)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer



import subprocess

cmd = 'echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE,shell=True).communicate()[0]).decode('utf-8')

import MeCab

mecab = MeCab.Tagger("-Ochasen")
mecab_neologd = MeCab.Tagger("-d {0} -Ochasen".format(path))

# ストップワードリストをインポートする
import urllib.request
url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
stopwords_text = os.path.join(data_dir,'/stopwords.txt')
urllib.request.urlretrieve(url, stopwords_text)
stopwords = []
with open(stopwords_text, 'r', encoding='utf-8') as f:
	stopwords = [w.strip() for w in f]

# 形態素解析して名詞だけ取り出すパターン
def tokenize2(text):
    node = mecab_neologd.parseToNode(text)
    word_list = []

    while node:
        if node.surface != '':
            elem = node.feature.split(',')
            if elem[0] == "名詞" :
                term = elem[6] if elem[6] != '*' else node.surface
                if term not in stopwords:
                    word_list.append(term)
        node = node.next

    return " ".join(word_list)

# 検索ワード、文章を入力してコサイン類似度上位20件の記事を検索する

def similar_article(word):

    sample_vector = vectorizer.transform([tokenize2(word)])

    # 計算した文書のtfidf_matrixと指定した文字列のベクトルのコサイン類似度を計算
    text_similarity = cosine_similarity(sample_vector, tfidf_matrix)
    similarity_df = pd.DataFrame(text_similarity)

    # 類似度上位20記事のindexとタイトルを出力
    title_list = []
    top_indices = np.argsort(-text_similarity)[0][:20]
    for index in top_indices:
        title_list.append({'index':index, 'title':article_df["title"][index], 'page_uniques': article_df['page_uniques'][index], 'date': article_df['date'][index]})
        list_df = pd.DataFrame(title_list)

    return list_df[['title','page_uniques','date']]



def main():
    # Streamlit が対応している任意のオブジェクトを可視化する (ここでは文字列)
    st.title('Recommend!')

    word = st.text_input(label='キーワードを入力してください', placeholder='キーワードを入力してください。')
    title = st.text_area(label='見出しを入力してください', placeholder='見出しを入力してください。', height=30)
    sentence = st.text_area(label='前文を入力してください', placeholder='前文を入力してください。', height=150)
    if st.button('検索'):
      if word:
        # 最後の試行で上のボタンがクリックされた
        st.write('入力済み')
        df_result = similar_article(word)
        st.write('検索結果')
        st.write(df_result)
      elif title:
        # 最後の試行で上のボタンがクリックされた
        st.write('入力済み')
        df_result2 = similar_article(title)
        st.write('検索結果')
        st.write(df_result2)
      elif sentence:
        # 最後の試行で上のボタンがクリックされた
        st.write('入力済み')
        df_result3 = similar_article(sentence)
        st.write('検索結果')
        st.write(df_result3)

if __name__ == '__main__':
    main()