import pandas as pd
import sklearn.model_selection as modelSelec
import sklearn.feature_extraction.text as featExtrac
import sklearn.linear_model as lm
import sklearn.metrics as metrics
import html2text as h2t
import tokenize as tkn
import warnings

def get_data():
    df = pd.read_csv("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/fake_or_real_news.csv")
    return df

def classifier(df):
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
    # print("accuracy: %0.3f" % score)
    confMat = metrics.confusion_matrix(y_test, pred, labels = ['FAKE', 'REAL'])
    # print confMat
    return linear_classif, tfidf_vectorizer

def main():
    df = get_data()
    classif, tfidf_vec = classifier(df)
    url = input("Enter a url: ")

    h = h2t.HTML2Text()
    h.ignore_links = True
    document = h.handle(url)

    tokens = document.split()

    tfidf = tfidf_vec.transform(tokens)

    pred = classif.predict(tfidf)

    print pred


if __name__ == "__main__":
    main()

