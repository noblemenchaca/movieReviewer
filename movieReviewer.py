import nltk, gzip
from nltk.corpus import stopwords

fp = 'mireviews.json.gz'
stops = set(stopwords.words('english'))
stops.remove('again')

def safe_div(a,b):
    result = -999
    if b != 0:
        result = float(a)/float(b)
    return result

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
      yield eval(l)

def document_features(document, wordsToSearchFor):
    features = {}
    documentWords = set(document)

    for word in wordsToSearchFor:
        features['countains({})'.format(word)] = (word in documentWords)
    return features

# create a list of review (dictionary) objects where review['reviewText'] is the text
# and review['overall'] is the rating

reviews = []
raw = ''
print('\nLoading corpus into dictionaries...')
for review in parse(fp):
    reviews.append(review)
    raw += review["reviewText"]

text = nltk.word_tokenize(raw)
lemmer = nltk.WordNetLemmatizer()
lemmas = [lemmer.lemmatize(t) for t in text if t not in stops]

print('\nLoading searchWords from frequency distribution...')
fdist = nltk.FreqDist(w.lower() for w in lemmas)
searchWords = [w for w in fdist.most_common(500)]

docs = []
print('\nParsing reviews into tuples...')
for review in reviews:
    rating = str(review["overall"])
    text = nltk.word_tokenize(review["reviewText"])
    words = [lemmer.lemmatize(t) for t in text]
    docs.append( (rating, words) )


print('\nMaking feature sets using searchWords...')
# Step 3: Generate features for documents
featureSets = [(document_features(doc, searchWords), c) for (c, doc) in docs]
# Step 4: Create train/test folds
trainSet, testSet = featureSets[:2000], featureSets[2000:]
print('\nSize of trainSet: ' + str(len(trainSet)))
print('Size of testSet: ' + str(len(testSet)))
# Step 5: Machine learning: Train classifier
print('\nTraining...')
classifier = nltk.NaiveBayesClassifier.train(trainSet)
# Step 6: Evaluate performance of machine learning classifier
performance = nltk.classify.accuracy(classifier, testSet)
print("\nAccuracy on test set: " + str(performance))
print(classifier.show_most_informative_features(25))

