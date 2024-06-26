import re
import io
import csv
import math
import tweepy
import string
import pandas as pd
from flask import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from preprocess import text_clean as tc, normalize as norm, case_folding as cf, stemmer as st, stop_remover as rs, hapus_kata as hk, tokenized as tk

app = Flask(__name__)
consumer_key = "2t5oIvUJUALUSQsv80GBPgzcK"
consumer_secret = "VTqeDj3cN6laPutp8Imcs1NFSCIu8519yoFtzr6MKyNQdBIBnL"
access_token = "1465978930381434884-yp0MSCbWKudobPIjA4UQ98e829JH4N"
access_secret = "vagmB3lEkhVEeMrg68JP025leESesy9XPuNujnjK4nB1R"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)
stemmer = StemmerFactory().create_stemmer(True)
stop_remover = StopWordRemoverFactory().create_stop_word_remover()
tfidf_scores = None
words = None
data = "C:/Users/HP/Documents/Berkas Penting/Tugas Kuliah/Semester 8/Tugas Akhir/CD/flask/dataset/dataset.csv"
vectorizer = None
classifier = None

# fungsi remove digunakan untuk menghapus karakter tertentu

def remove(text):
    text = re.sub('[0-9]+', '', str(text))
    text = re.sub(r'\$\w*', '', str(text))
    text = re.sub(r'rt :[\s]+', '', str(text))
    text = re.sub("b'|b\"", '', str(text))
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", str(text))
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub('<.*?>', '', str(text))
    text = re.sub("\n", " ", str(text))
    text = re.sub(r"\b[a-zA-Z]\b", "", str(text))
    text = re.sub(r',', '', str(text))
    text = re.sub(r'#', '', str(text))
    # text = ' '.join(text.split())
    text = text.encode('ascii', 'replace').decode('ascii')
    return text

# tokenizing memisah tiap kalimat menjadi beberapa pecahan kata

def tokenizing(text):
    x = text.split()
    return x

# Stopword menghapus kata yang tidak mengandung arti

def stopword(text, stop_remover):
    text = stop_remover.remove(text)
    return text


# Stemmer menghapus kata yang memiliki imbuhan
def stemming(text, stemmer):
    text = stemmer.stem(str(text))
    return text


# Normasisasi kata
norm = {'mahasi': 'mahasiswa', 'penga': 'Pengawasan ', 'kek': ' kayak ', 'dr.': 'dokter', ' udah ': ' sudah ', 'sdh ': 'sudah ', 'mw ': 'mau',
        ' ga ': 'tidak', 'yg ': 'yang ', 'tak ': 'tidak', 'hrs ': 'harus', 'bp ': 'bapak', 'krn': 'karena', 'dlm': 'dalam', ' dr ': 'dari',
        'kpd': 'kepada', 'klo': 'kalau', 'tdk': 'tidak', 'dgn': 'dengan', 'dg': 'dengan', 'trus': 'terus', 'bwh': 'bawah', 'tsb': 'tersebut',
        'tp': 'tapi', 'tpi': 'tapi', 'bkn': 'bukan', 'ttg ': 'tentang', 'gnt ': 'ganti', 'bhwa': 'bahwa', 'skrg': 'sekarang',
        ' ama ': ' sama', 'bnyk': 'banyak', 'gmn': 'bagaimana', 'dr. ': 'dokter', ' mo ': 'mau', 'atw ': 'atau', ' ri ': ' republik indonesia ', 'sbg ': 'sebagai', 'Yg': 'Yang'}


def normalize(str_text):
    for i in range(len(str_text)):
        for a in norm:
            str_text[i] = str_text[i].replace(a, norm[a])
    return str_text


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/scrap_data")
def crawling():
    return render_template("scrap_data.html")


@app.route('/download_data', methods=["POST"])
def download_csv():
    query = request.form['query']  # Masukkan query untuk download data
    count = int(request.form['count'])  # Jumlah data untuk download data
    tweets = api.search_tweets(
        q=query, count=count, tweet_mode='extended')
    csv_data = []
    for tweet in tweets:  # untuk setiap tweet di dalam tweets, tambahkan variabel csv_data berupa date, username, dan teks
        csv_data.append({
            'date': tweet.created_at,
            'username': tweet.user.name,
            'text': tweet.full_text.encode('utf-8')
        })
    headers = {
        'Content-Disposition': 'attachment; filename=downloaded_data.csv',
        'Content-Type': 'text/csv'
    }
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=['date', 'username', 'text'])
    writer.writeheader()
    writer.writerows(csv_data)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers=headers
    )


@app.route("/process", methods=["GET", "POST"])
def preprocess():
    global df
    if request.method == 'GET':
        return render_template('process.html')
    elif request.method == 'POST':
        global words
        csv_file = request.files.get('file')
        df = pd.read_csv(csv_file, sep=';')
        df['Case Folding'] = df['text'].apply(
            lambda x: x.lower() if isinstance(x, str) else x)
        df['Text Cleaning'] = df['Case Folding'].apply(lambda x: remove(x))
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df['Tokenizing'] = df['Text Cleaning'].apply(tokenizing)
        df['Normalize Text'] = df['Tokenizing'].apply(normalize)
        df['Stemming'] = df['Normalize Text'].apply(stemming, stemmer=stemmer)
        df['Stopword'] = df['Stemming'].apply(
            stopword, stop_remover=stop_remover)

        words = df['Tokenizing'].apply(pd.Series).stack().unique()
        df['Sentiment Class'] = df['sentiment'].apply(
            lambda x: 'Positif' if x == 1 else ('Negatif' if x == -1 else 'Netral'))

        # Hitung Term Frequency (TF) untuk tiap kata, tiap kelas sentimen
        tf_dict = {}
        for sentiment_class in ['Positif', 'Negatif', 'Netral']:
            df_class = df[df['Sentiment Class'] == sentiment_class]
            words_class = df_class['Normalize Text'].apply(
                pd.Series).stack().unique()
            for word in words_class:
                tf_dict[(sentiment_class, word)] = max(row.count(word)
                                                       for row in df_class['Normalize Text'])

        # Menghitung Inverse Document Frequency (IDF) untuk setiap kata
        num_docs = len(df)
        idf_dict = {}
        for word in words:
            num_docs_with_word = sum(
                1 for row in df['Normalize Text'] if word in row)
            if num_docs_with_word > 0:
                idf_dict[word] = math.log(num_docs / num_docs_with_word)

        # Hitung skor TF-IDF untuk setiap kata, tiap kelas sentimen
        tfidf_dict = {}
        for sentiment_class in ['Positif', 'Negatif', 'Netral']:
            df_class = df[df['Sentiment Class'] == sentiment_class]
            words_class = pd.DataFrame(
                df_class['Normalize Text'].apply(pd.Series)).stack().unique()

            for word in words_class:
                tfidf_dict[(sentiment_class, word)] = tf_dict[(
                    sentiment_class, word)] * idf_dict[word]

        # Simpan hasil TF-IDF dalam variabel global
        global tfidf_scores
        tfidf_scores = pd.DataFrame({
            'Sentiment Class': [sentiment_class for sentiment_class, word in tfidf_dict.keys()],
            'Word': [word for sentiment_class, word in tfidf_dict.keys()],
            'TF': [tf_dict[(sentiment_class, word)] for sentiment_class, word in tfidf_dict.keys()],
            'IDF': [idf_dict[word] for sentiment_class, word in tfidf_dict.keys()],
            'TF-IDF': [tfidf_dict[(sentiment_class, word)] for sentiment_class, word in tfidf_dict.keys()]
        })

        table_html = df.to_html(classes='my-table', index=False)
        return render_template('process.html', table_html=table_html)


@app.route('/tfidf', methods=["GET", "POST"])
def hitung_tfidf():
    global df
    global tfidf_scores
    if tfidf_scores is not None:
        # Reset the index of tfidf_scores
        tfidf_scores = tfidf_scores.reset_index(drop=True)
        # Filter tfidf_scores berdasarkan sentiment class
        positive_tfidf_scores = tfidf_scores[tfidf_scores['Sentiment Class'] ==
                                             'Positif'].loc[tfidf_scores[tfidf_scores['Sentiment Class'] == 'Positif'].index]
        negative_tfidf_scores = tfidf_scores[tfidf_scores['Sentiment Class'] ==
                                             'Negatif'].loc[tfidf_scores[tfidf_scores['Sentiment Class'] == 'Negatif'].index]
        neutral_tfidf_scores = tfidf_scores[tfidf_scores['Sentiment Class'] ==
                                            'Netral'].loc[tfidf_scores[tfidf_scores['Sentiment Class'] == 'Netral'].index]

        # Render the template with the separate tables for each sentiment class
        positive_table_html = positive_tfidf_scores.to_html(
            classes='my-table', index=False)
        negative_table_html = negative_tfidf_scores.to_html(
            classes='my-table', index=False)
        neutral_table_html = neutral_tfidf_scores.to_html(
            classes='my-table', index=False)
        return render_template('tfidf.html', positive_table_html=positive_table_html, negative_table_html=negative_table_html, neutral_table_html=neutral_table_html)
    else:
        return "No TF-IDF scores available"


@app.route('/naive_bayes', methods=["POST", "GET"])
def tfidf():
    global vectorizer
    global classifier
    df = pd.read_csv(data, sep=';')
    global vectorizer
    global classifier
    df = pd.read_csv(data, sep=';')
    df['Case Folding'] = df['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['Text Cleaning'] = df['Case Folding'].apply(lambda x: remove(x))
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['Tokenizing'] = df['Text Cleaning'].apply(tokenizing)
    df['Normalize Text'] = df['Tokenizing'].apply(normalize)
    table_html = df.to_html(classes='my-table', index=False)
    X = df['Normalize Text']
    y = df['sentiment']
    test_size = 0.1
    random_state = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train = [' '.join(tokens) for tokens in X_train]
    X_test = [' '.join(tokens) for tokens in X_test]
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(
        y_test, y_pred, average='macro') * 100, 2)
    recall = round(recall_score(y_test, y_pred, average='macro') * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='macro') * 100, 2)
    return render_template('nb.html', table_html=table_html, accuracy=accuracy, precision=precision, recall=recall, f1=f1, random_state=random_state, test_size=test_size)


@app.route('/input_param', methods=["POST", "GET"])
def input_param():
    global vectorizer
    global classifier
    df = pd.read_csv(data, sep=';')
    df['text_clean'] = df['text'].apply(lambda x: tk(x))
    df['normalize_text'] = df['text_clean'].apply(lambda x: norm(x))
    df['case_folding'] = df['normalize_text'].apply(lambda x: cf(x))
    df['stem_text'] = df['case_folding'].apply(lambda x: st(x))
    df['stopword'] = df['stem_text'].apply(lambda x: rs(x))
    df['hapus_kata'] = df['stopword'].apply(lambda x: hk(x))
    df['stopword'] = df['tokenized'].apply(lambda x: tk(x))
    df['tokenized_text'] = df['tokenized'].apply(lambda x: ' '.join(x))
    table_html = df.to_html(classes='my-table', index=False)
    X = df['tokenized_text']
    y = df['sentiment']
    test_size = float(request.form['test_size'])
    random_state = int(request.form['random_state'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    X_train = [' '.join(tokens) for tokens in X_train]
    X_test = [' '.join(tokens) for tokens in X_test]
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(
        y_test, y_pred, average='macro') * 100, 2)
    recall = round(recall_score(y_test, y_pred, average='macro') * 100, 2)
    f1 = round(f1_score(y_test, y_pred, average='macro') * 100, 2)
    return render_template('nb.html', table_html=table_html, accuracy=accuracy, precision=precision, recall=recall, f1=f1, random_state=random_state, test_size=test_size)


@app.route("/predict", methods=["POST", "GET"])
def predict():
    global vectorizer
    global classifier
    text = request.form['text']
    input_text = stemming(tokenizing(remove(text.lower())), stemmer=stemmer)
    input_text = ' '.join(input_text)
    input_tfidf = vectorizer.transform([input_text])
    prediction = classifier.predict(input_tfidf)[0]
    return render_template('nb.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
