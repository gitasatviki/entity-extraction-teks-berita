import pickle
import stanza
from gensim.models import FastText 
from collections import OrderedDict

fasttext_model = FastText.load('fasttext_compatible.bin', mmap='r')
fasttext_wv = fasttext_model.wv

with open('crf_model.pkl', 'rb') as f:
    crf_model = pickle.load(f)

nlp = stanza.Pipeline(
    lang='id',
    processors='tokenize,pos',
    verbose=False
)

def word2features(sent, i, fasttext_wv):
    word = sent[i][0]
    postag = sent[i][1]

    common_prefixes = [
        'me', 'di', 'ber', 'ter', 'pe', 'per', 'ke', 'se',
        'be', 'te', 'pen', 'peng', 'pem'
    ]
    common_suffixes = [
        'kan', 'an', 'i', 'nya', 'lah', 'kah', 'tah',
        'pun', 'ku', 'mu', 'nya'
    ]

    features = OrderedDict()

    features['bias'] = 1.0
    features['word.lower()'] = word.lower()
    features['word.istitle()'] = word.istitle()
    features['word.isupper()'] = word.isupper()
    features['word.isdigit()'] = word.isdigit()
    features['word.len'] = len(word)
    features['postag'] = postag
    features['postag[:2]'] = postag[:2]

    for l in range(2, 5):
        if len(word) >= l:
            features[f'prefix_{l}'] = word[:l].lower()
            features[f'suffix_{l}'] = word[-l:].lower()

    lower_word = word.lower()
    for pref in common_prefixes:
        if lower_word.startswith(pref):
            features[f'has_prefix_{pref}'] = True

    for suff in common_suffixes:
        if lower_word.endswith(suff):
            features[f'has_suffix_{suff}'] = True

    features['has_hyphen'] = '-' in word

    try:
        embedding = fasttext_wv[word.lower()]
        for idx, val in enumerate(embedding):
            features[f'ft_{idx}'] = float(val)
    except KeyError:
        for idx in range(fasttext_wv.vector_size):
            features[f'ft_{idx}'] = 0.0

    if i > 0:
        prev_word = sent[i - 1][0]
        prev_pos = sent[i - 1][1]
        features['-1:word.lower()'] = prev_word.lower()
        features['-1:word.istitle()'] = prev_word.istitle()
        features['-1:word.isupper()'] = prev_word.isupper()
        features['-1:postag'] = prev_pos
        features['-1:postag[:2]'] = prev_pos[:2]
        for l in range(2, 4):
            if len(prev_word) >= l:
                features[f'-1:prefix_{l}'] = prev_word[:l].lower()
                features[f'-1:suffix_{l}'] = prev_word[-l:].lower()
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        next_word = sent[i + 1][0]
        next_pos = sent[i + 1][1]
        features['+1:word.lower()'] = next_word.lower()
        features['+1:word.istitle()'] = next_word.istitle()
        features['+1:word.isupper()'] = next_word.isupper()
        features['+1:postag'] = next_pos
        features['+1:postag[:2]'] = next_pos[:2]
        for l in range(2, 4):
            if len(next_word) >= l:
                features[f'+1:prefix_{l}'] = next_word[:l].lower()
                features[f'+1:suffix_{l}'] = next_word[-l:].lower()
    else:
        features['EOS'] = True  # 

    return features

def sent2features(sent, fasttext_wv):
    return [word2features(sent, i, fasttext_wv) for i in range(len(sent))]

def predict_entities(text):
    doc = nlp(text)
    results = []

    for sent in doc.sentences:
        sentence = [(word.text, word.upos, 'O') for word in sent.words]

        if not sentence:
            continue

        features = sent2features(sentence, fasttext_wv)
        predictions = crf_model.predict([features])[0]

        for (token, pos, _), label in zip(sentence, predictions):
            results.append({
                'token': token,
                'pos': pos,
                'entity': label
            })

    return results