import io
import csv
import tweepy
import pandas as pd
import configparser
from flask import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from preprocess import remove, tokenizing, stopword, stemming, normalize
from tfidf import compute_idf, compute_tf, compute_tfidf

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('config.ini')
auth = tweepy.OAuthHandler(config['API_KEYS']['api_key'], config['API_KEYS']['api_secret'])
auth.set_access_token(config['API_KEYS']['access_token'], config['API_KEYS']['access_secret'])
api = tweepy.API(auth)
stemmer = StemmerFactory().create_stemmer(True)
stop_remover = StopWordRemoverFactory().create_stop_word_remover()
tfidf_scores = None
words = None
data = "./dataset.csv"
vectorizer = None
classifier = None
table_html = None
result_df = None


@app.route("/")
def home():
    title = 'Home'
    return render_template('index.html', title=title)


@app.route("/scrap_data")
def crawling():
    title = 'Download data'
    return render_template("download.html", title=title)


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
    title = 'Pre-processing'
    global df, table_html, result_df
    if request.method == 'GET':
        return render_template('preprocess.html')
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
        
        # Filtering sentiment classes
        positive_df = df[df['sentiment'] == 1]
        negative_df = df[df['sentiment'] == -1]
        neutral_df = df[df['sentiment'] == 0]
        
        sentiment_classes = {1: positive_df, -1: negative_df, 0: neutral_df}
        
        result = []
        for sentiment, class_df in sentiment_classes.items():
            word_set = set()
            docs = []
            for doc in class_df['text']:
                bow = doc.split()
                word_dict = dict.fromkeys(bow, 0)
                for word in bow:
                    word_dict[word] += 1
                docs.append(word_dict)
                word_set.update(word_dict.keys())
            
            idfs = compute_idf(docs)
            for doc in class_df['text']:
                bow = doc.split()
                word_dict = dict.fromkeys(bow, 0)
                for word in bow:
                    word_dict[word] += 1
                tf = compute_tf(word_dict, bow)
                tfidf = compute_tfidf(tf, idfs)
                for word in tfidf:
                    result.append([word, tf[word], idfs[word], tfidf[word], sentiment])    
        result_df = pd.DataFrame(result, columns=['Word', 'TF', 'IDF', 'TF-IDF', 'sentiment class'])        
        table_html = df.to_html(classes='table table-hover', index=False)        
        return redirect(url_for('result', title=title))

@app.route('/result', methods=["GET", "POST"])
def result():
    global table_html
    return render_template('processed.html', table_html=table_html)

@app.route('/tfidf', methods=["GET", "POST"])
def hitung_tfidf():
    global result_df
    if result_df is not None:
        return render_template('tfidf.html', tables=[result_df.to_html(classes='table table-hover')], titles=result_df.columns.values)
    else:
        return "No TF-IDF scores available"


@app.route('/naive_bayes', methods=["POST", "GET"])
def tfidf():
    title = 'Naive Bayes'
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
    return render_template('nb.html', table_html=table_html, accuracy=accuracy, precision=precision, recall=recall, f1=f1, random_state=random_state, test_size=test_size, title=title)


@app.route('/input_param', methods=["POST", "GET"])
def input_param():
    global vectorizer
    global classifier
    df = pd.read_csv(data, sep=';')
    df['Case Folding'] = df['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    df['Text Cleaning'] = df['Case Folding'].apply(lambda x: remove(x))
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['Tokenizing'] = df['Text Cleaning'].apply(lambda x: tokenizing(x))
    df['Normalize Text'] = df['Tokenizing'].apply(lambda x: normalize(x))
    table_html = df.to_html(classes='my-table', index=False)
    X = df['Normalize Text']
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
