    query = DataModel.objects.all()
    res = []
    textDF = pandas.DataFrame()

    labels, texts = [], []

    for row in query:
        texts.append(row.text)
        labels.append(row.label)

    textDF['text'] = texts
    textDF['label'] = labels
    print(textDF.shape)

    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing

    X_train, X_test, y_train, y_test = train_test_split(textDF.text, textDF.label, test_size=0.25, random_state=23)
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    # X_train = X_train.map(lambda x: tokenizer_yh(x))
    # X_test = X_test.map(lambda x: tokenizer_yh(x))

    from sklearn.feature_extraction.text import TfidfVectorizer

    # vec_tfidf = TfidfVectorizer(ngram_range =(1,1),tokenizer=tokenizer_yh, min_df = 15, max_df = 0.9)
    print("TF-IDF")
    vec_tfidf = TfidfVectorizer()
    vec_tfidf_f = vec_tfidf.fit(X_train)
    train_dtm_ngram = vec_tfidf_f.transform(X_train).toarray()
    test_dtm_ngram = vec_tfidf_f.transform(X_test).toarray()
    # print(train_dtm_ngram)

    print("标签向量化")
    # label encode theJ target variable
    encoder = preprocessing.LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.fit_transform(y_test)

    # print(y_train.shape)

    '''
    使用xgboost进行分类测试
    '''
    # from xgboost import XGBClassifier
    # xgbc = XGBClassifier(n_estimators=200)
    # xgbc_ngram = xgbc.fit(train_dtm_ngram, y_train)
    # y_predict = xgbc_ngram.predict(test_dtm_ngram)
    # print(y_predict.shape)

    # from sklearn.metrics import accuracy_score, classification_report
    # print(accuracy_score(y_predict, y_test))
    # print(classification_report(y_predict, y_test))

    '''
    使用jEnsemble进行分类测试
    '''
    E = Ensemble(H=10, blocksize=100)
    a = train_dtm_ngram[0]
    # b = train_dtm_ngram[1]
    # print(E.cos_dist(a, b))

    ######获取测试数据
    f = E.gen_blocks(test_dtm_ngram, y_test)
    block = next(f)
    test_x = block.getX()
    test_y = block.getY()

    # print(test_x)
    # print(block.getY())
    # clt = E.make_basic_model(block)
    # print(clt.predict([a]))

    E.fit(train_dtm_ngram, y_train)
    y_pred = E.predict(test_x)

    print(y_pred)
    print(test_y)

    from sklearn import metrics
    from sklearn.metrics import classification_report

    recall = metrics.recall_score(test_y, y_pred)
    F1 = metrics.f1_score(test_y, y_pred)
    print("正确率：", np.mean(y_pred == test_y))
    print("召回率：", recall)
    print("F1：", F1)

    target_names = ['obama', 'smartphone']
    print(classification_report(test_y, y_pred, target_names=target_names))
