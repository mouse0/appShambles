from flask import Flask, redirect, render_template, request

import pandas as pd
import sklearn.model_selection as modelSelec
import sklearn.feature_extraction.text as featExtrac
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import html2text as h2t

app = Flask(__name__)


@app.route('/')
def homepage():
    # Return a Jinja2 HTML template and pass in image_entities as a parameter.
    return render_template('homepage.html')

@app.route('/run_language', methods=['GET', 'POST'])
def get_data():
    df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")
    return df

def getClassifier():
    df = get_data()
    y = df.label
    df.drop("label", axis=1)
    X_train, X_test, y_train, y_test = modelSelec.train_test_split(df['text'], y,
                                                                test_size=0.33,
                                                                random_state=53)
    tfidf_vectorizer = featExtrac.TfidfVectorizer(stop_words='english', max_df=0.7)
    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    linear_classif = lm.PassiveAggressiveClassifier(n_iter=50)
    linear_classif.fit(tfidf_train, y_train)
    pred = linear_classif.predict(tfidf_test)
    score = metrics.accuracy_score(y_test, pred)
    confMat = metrics.confusion_matrix(y_test, pred, labels = ['FAKE', 'REAL'])
    return linear_classif

def homepage():
    text = request.form['text']
    classif = getClassifier()

    h = h2t.HTML2Text()
    h.ignore_links = TRUE

    url = request.form['text']
    document = h.handle(url)

    result = classif.predict(document)

    return render_template('homepage.html', text=text, entities=result)

@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
