import re

import spacy
import textacy

nlp = spacy.load('en')
entities_to_remove = ['PERSON', 'GPE', 'LOC', 'NORP', 'FACILITY', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LANGUAGE', 'DATE',
                      'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

# Only nouns, verbs and adjectives will be kept.
tags_to_keep = ['JJ', 'NN', 'NNP', 'NNPS', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

#https://gist.github.com/AndradeEduardo/ab0dfd658ca9dbe234a4148471a6ba87#file-lda_modelling-ipynb
#https://gist.github.com/AndradeEduardo/ab0dfd658ca9dbe234a4148471a6ba87#file-lda_modelling-ipynb


def clean_single_doc(sentence):
    # sentence = re.sub(r"\\\'", r' ', sentence)
    # sentence = re.sub(r' s ', r" ", sentence)
    ## removing character sequences of Word files
    # sentence = re.sub(r'(\\n|\\xe2|\\xa2|\\x80|\\x9c|\\x9c|\\x9d|\\t|\\nr|\\x93)', r' ', sentence)
    # sentence = sentence.replace(r"b'", ' ')  ## removing the beginning of some documents
    # sentence = re.sub(r'\{[^\}]+\}', r' ', sentence)  ## removing word macros

    ## removing emails
    sentence = textacy.preprocess.replace_emails(sentence, replace_with=r' ')

    ## removing urls
    sentence = textacy.preprocess.replace_urls(sentence, replace_with=r' ')

    # sentence = re.sub(r'(#|$| [b-z] |\s[B-Z]\s|\sxxx\s|\sXXX\s|XXX\w+)', r' ', sentence)

    ## removing character sequences of pdf files
    # sentence = re.sub(r'(\\x01|\\x0c|\\x98|\\x99|\\xa6|\\xc2|\\xa0|\\xa9|\\x82)', r' ', sentence)

    # remove address
    # sentence = re.sub(r'(c/-d*)', r' ', sentence)

    sentence = re.sub(r'(\\x01|\\x0c|\\x98|\\x99|\\xa6|\\xc2|\\xa0|\\xa9|\\x82|\\xb7)', r' ', sentence)

    # removing trade mark of specific pdf file
    sentence = sentence.replace(r'LE G A SA L E M D PL O C E S', ' ')

    # striping consecutive white spaces
    sentence = re.sub(r'\s\s+', ' ', sentence).strip()
    sentence = sentence.strip()
    return sentence


def filter_single_nlp_doc(doc):
    filtered_doc = ''
    for sentence in doc.sents:
        sent_filt_text = ' '.join(
            [token.lemma_ for token in sentence if
             (token.tag_ in tags_to_keep and not token.is_stop and not token.ent_type_ in entities_to_remove)])
        filtered_doc = filtered_doc + ' ' + sent_filt_text

    return filtered_doc

# Method use pos_
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out