from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Tokenizer
import pandas as pd
import pickle
from tqdm import tqdm
from transformers import BertTokenizerFast
import logging
import numpy as np

corpus = "./Datas/chatbot.txt"
save_corpus = "./Datas/chatbot_token.pkl"
vocab_path = "./Datas/vocab.txt"


def pre_process():

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id

    # 读取训练数据集
    with open(corpus, 'rb') as f:
        data = f.read().decode("utf-8")
        train_data = data.split("\r\n\r\n")

    del(train_data[-1])
    # 开始进行tokenize
    # 保存所有的对话数据,每条数据的格式为："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    dialogue_len = []  # 记录所有对话tokenize之后的长度，用于统计中位数与均值
    dialogue_list = []
    with open(save_corpus, "w", encoding="utf-8") as f:
        for index, dialogue in enumerate(tqdm(train_data)):
            if "\r\n" in data:
                utterances = dialogue.split("\r\n")
            else:
                utterances = dialogue.split("\n")

            input_ids = [cls_id]
            for utterance in utterances:
                input_ids += tokenizer.encode(utterance, add_special_tokens=False)
                input_ids.append(sep_id)
            dialogue_len.append(len(input_ids))
            dialogue_list.append(input_ids)

    len_mean = np.mean(dialogue_len)
    len_median = np.median(dialogue_len)
    len_max = np.max(dialogue_len)
    with open(save_corpus, "wb") as f:
        pickle.dump(dialogue_list, f)


if __name__ == '__main__':
    pre_process()
