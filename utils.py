import MeCab
import sentencepiece
import copy
import sentencepiece as spm
from tqdm import tqdm_notebook
import numpy as np
from collections import OrderedDict
import mojimoji
import re




pattern_tapple = ((r'[一二三四五六七八九十〇]+号', r'一号'), \
                  (r'平成[一二三四五六七八九十〇]+年', r'平成一年'), \
                  (r'[一二三四五六七八九十〇]+月', r'一月'), \
                  (r'(午[(前)|(後)])[一二三四五六七八九十〇]+時',r'午前一時'), \
                  (r'[一二三四五六七八九十〇]+月[一二三四五六七八九十〇]+日', r'一月一日'),\
                  (r'^[一二三四五六七八九十壱弐参拾百千万萬億兆〇]$', r'一'),\
                  (r'^[0-9]+$', r'1'), \
                  (r'^[0-9]+(|名|号|分|円|番|発|戦|年度|種|期|年|月|日|時|)$', r'1\1'), \
                  (r'^[一二三四五六七八九十〇]+(|名|号|分|円|番|発|戦|年度|種|期|年|月|日|時|)$', r'1\1') \


    )
inflection_tapple = ('動詞', '形容詞', '助動詞')
def translate_word(ma_word, stopwords = []):
    word = ma_word[-3] if ma_word[1] in inflection_tapple else ma_word[0]
    word = mojimoji.zen_to_han(word, kana=False)
    #stopwordsが定義された場合の対応
    if stopwords and word in stopwords:
        word = ''
        return word

    if ma_word[2] == '固有名詞' and ma_word[3] == '人名':
        word = '佐村河内守'
        return word

    for patterns in pattern_tapple:
        word = re.sub(patterns[0], patterns[1], word)

    return word


def dalete_text_ma_column(df_train, df_test):
    try:
        df_train = df_train.drop('speech_text_ma', axis=1)
        df_test  = df_test.drop('speech_text_ma', axis=1)
    except KeyError:
        pass

    return df_train, df_test

##########Morphological analysis with MeCab##########↓
class Mecab_Analysis(object):
    def __init__(self, dic_path=''):
        try:
            if dic_path:
                dic_path = '-d %s' % dic_path
                self.m = MeCab.Tagger(dic_path)
            else:
                self.m = MeCab.Tagger()
        except RuntimeError:
            print('Invalid dictionary path. Use the default dictionary.')
            self.m = MeCab.Tagger()

        self.m.parse('')


    def Morphological_Analysis(self, text):
        result_list = []
        m_texts = self.m.parse(text)
        m_texts = m_texts.split('\n')
        for m_text in m_texts:
            m_text = m_text.split('\t')
            word = m_text[0]
            if word == 'EOS':
                break

            pos = m_text[1]
            pos = pos.split(',')
            append_list = [word] + pos
            result_list.append(copy.copy(append_list))

        return result_list
##########Morphological analysis with MeCab##########↑


def create_chunk_dataset(df, chunk_size=5):
    assert chunk_size % 2 == 1, 'chunk_size should be odd number.'
    labels = df['committee'].values.tolist()
    texts = df['speech_text_ma'].values.tolist()
    chunk_mean = chunk_size // 2
    df_len = len(df)
    #先頭のラベルをdfと合わせるために先頭にchunk_meanの数だけ末尾のindexの要素を加える
    #末尾も同様に先頭のindexを加える
    texts = [texts[idx] for idx in range(chunk_mean*(-1), 0)] + \
            texts + \
            [texts[idx] for idx in range(0, chunk_mean+1)]
    #先頭に加えた末尾を捨てる
    result_texts = []
    #真ん中のword頻度を最大に端に行けば行くほど頻度を少なくする
    for idx in range(0, df_len):
        words = []
        for jdx in range(idx, idx+chunk_size):
            origin_point = jdx - idx
            word_freq = chunk_mean - abs(chunk_mean - origin_point)
            words.extend([texts[jdx] for _ in range(0, word_freq+1)])

        result_texts.append(' '.join(words))

    return result_texts, labels
