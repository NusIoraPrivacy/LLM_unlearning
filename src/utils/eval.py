
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import argparse
from fuzzywuzzy import fuzz

from sentence_transformers import SentenceTransformer

def extract_word_ratio(origin_sentence, rslt_sentence):
    original_words = origin_sentence.split()
    rslt_words = rslt_sentence.split()
    original_word_set = set(original_words)
    rslt_words_set = set(rslt_words)
    extract_count = len(rslt_words_set.symmetric_difference(original_word_set))
    extract_ratio = extract_count / len(rslt_words_set)
    return extract_ratio


def extract_sentence_similarity(origin_sentence, rslt_sentence):

    st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    original_embeddings = st_model.encode(origin_sentence)
    extract_embeddings = st_model.encode(rslt_sentence)

    similarity_score = np.dot(original_embeddings, extract_embeddings.T)
    return similarity_score

def has_common_subphrase(s1, s2, min_words=1):
    words1 = s1.split()

    for i in range(len(words1)):
        for j in range(i + min_words, len(words1) + 1):
            subphrase = ' '.join(words1[i:j])
            if subphrase in s2:
                print(subphrase)
                return subphrase
    return False

def judge_extract_sucess(args, step, logger, summarize_rslt, 
                         origin_senetence, extract_sentence, 
                         min_verbatim_words):
    # match
    similarity = fuzz.ratio(origin_senetence, extract_sentence)
    common_subphrase = has_common_subphrase(origin_senetence, extract_sentence, min_words=min_verbatim_words)
    
    # judge match
    if similarity == 100:
        # if input the summary, skip the same
        if args.summary_flag:
            if extract_sentence in summarize_rslt:
                return False

        print(f'[Same]: (origin abstract: {origin_senetence}, extract abstract: {extract_sentence})')
        logger.info(f'[Step]:{step},[Same]: (same abstract: {extract_sentence})')
        return True

    elif common_subphrase is not False:
        # if input the summary, skip the same
        if args.summary_flag:
            if has_common_subphrase(extract_sentence, summarize_rslt):
                return False

        print(f'[COMMON SUBSTRING]: (origin abstract: {origin_senetence}, extract abstract: {extract_sentence}, common substring: {common_subphrase})')
        logger.info(f'[Step]:{step}, [COMMON SUBSTRING]: (common substring: {common_subphrase})')
        return True

    elif similarity > 80:
        print(f'[SIMILAR]: (origin abstract: {origin_senetence}, extract abstract: {extract_sentence})')
        logger.info(f'[Step]:{step},[SIMILAR]: (origin abstract: {origin_senetence}, extract abstract: {extract_sentence})')
        return True
    
    else:
        return False