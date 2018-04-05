#!/usr/bin/env python3
# coding: utf-8
# File: mi.py
# Author: lhy<lhy_in_blcu@126.com,https://huangyong.github.io>
# Date: 18-3-26

import collections
import math
import jieba.posseg as pseg

class MI_Train:
    def __init__(self, window_size, filepath, mipath):
        self.window_size = window_size
        self.filepath = filepath
        self.mipath = mipath

    #对语料进行处理
    def build_corpus(self):
        def cut_words(sent):
            return [word.word for word in pseg.cut(sent) if word.flag[0] not in ['x', 'w', 'p', 'u', 'c']]
       # sentences = [sent.split(' ') for sent in open(self.filepath).read().split('\n')]，若处理英文语料则使用这种方法
        sentences = [cut_words(sent) for sent in open(self.filepath).read().split('\n')]
        return sentences

    #统计词频
    def count_words(self, sentences):
        words_all = list()
        for sent in sentences:
            words_all.extend(sent)
        word_dict = {item[0]:item[1] for item in collections.Counter(words_all).most_common()}

        return word_dict, len(words_all)
    #读取训练语料
    def build_cowords(self, sentences):
        train_data = list()
        for sent in sentences:
            for index, word in enumerate(sent):
                if index < self.window_size:
                    left = sent[:index]
                else:
                    left = sent[index - self.window_size: index]
                if index + self.window_size > len(sent):
                    right = sent[index+1 :]
                else:
                    right = sent[index+1: index + self.window_size + 1]
                data = left + right + [sent[index]]

                if '' in data:
                    data.remove('')
                train_data.append(data)
        return train_data

    #统计共现次数
    def count_cowords(self, train_data):
        co_dict = dict()
        print(len(train_data))
        for index, data in enumerate(train_data):
            for index_pre in range(len(data)):
                for index_post in range(len(data)):
                    if data[index_pre] not in co_dict:
                        co_dict[data[index_pre]] = data[index_post]
                    else:
                        co_dict[data[index_pre]] += '@' + data[index_post]
        return co_dict

    # 计算互信息
    def compute_mi(self, word_dict, co_dict, sum_tf):
        def compute_mi(p1, p2, p12):
            return math.log2(p12) - math.log2(p1) - math.log2(p2)

        def build_dict(words):
            return {item[0]:item[1] for item in collections.Counter(words).most_common()}

        mis_dict = dict()
        for word, co_words in co_dict.items():
            co_word_dict = build_dict(co_words.split('@'))
            mi_dict = {}
            for co_word, co_tf in co_word_dict.items():
                if co_word == word:
                    continue
                p1 = word_dict[word]/sum_tf
                p2 = word_dict[co_word]/sum_tf
                p12 = co_tf/sum_tf
                mi = compute_mi(p1, p2, p12)
                mi_dict[co_word] = mi
            mi_dict = sorted(mi_dict.items(), key = lambda asd:asd[1], reverse= True)
            mis_dict[word] = mi_dict

        return mis_dict

    # 保存互信息文件
    def save_mi(self, mis_dict):
        f = open(self.mipath, 'w+')
        for word, co_words in mis_dict.items():
            co_infos = [item[0] + '@' + str(item[1]) for item in co_words]
            f.write(word + '\t' + ','.join(co_infos) + '\n')
        f.close()

    # 运行主函数
    def mi_main(self):
        print('step 1/6: build corpus ..........')
        sentences = self.build_corpus()
        print('step 2/6: compute worddict..........')
        word_dict, sum_tf = self.count_words(sentences)
        print('step 3/6: build cowords..........')
        train_data = self.build_cowords(sentences)
        print('step 4/6: compute coinfos..........')
        co_dict = self.count_cowords(train_data)
        print('step 5/6: compute words mi..........')
        mi_data = self.compute_mi(word_dict, co_dict, sum_tf)
        print('step 6/6: save words mi..........')
        self.save_mi(mi_data)
        print('done!.......')

#测试
def test():
    filepath = './data/data.txt'
    mipath = './data/result.txt'
    window_size = 5
    mier = MI_Train(window_size, filepath, mipath)
    mier.mi_main()

if __name__=='__main__':
    test()



