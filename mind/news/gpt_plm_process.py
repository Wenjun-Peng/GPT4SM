import numpy as np
from news.utils import sent_tokenize
from utils import MODEL_CLASSES
from .gpt_process import GPTInfo

class PLMGPTInfo(GPTInfo):
    def __init__(self, args, emb_file, news_file):
        self.args = args
        self.news_file = news_file
        self.emb_file = emb_file
        self.nid2index = {}

        self.news_raw_title = {}
        self.news = {}

        self.record_size = 8 + 1536 * 4 * 2

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.tokenizers = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=True)

    def process(self):
        self.process_news()
        self.process_emb()
        self.process_title_index()

    def process_news(self):
        with open(self.news_file, 'r', encoding="utf8") as f:
            for line in f:
                docid, title, body = line.strip('\n').split('\t')
                docid = int(docid[1:])
                self.update_dict(self.nid2index, docid)

                self.update_dict(self.news_raw_title, docid, title)

                title_index = sent_tokenize(self.tokenizers, title, self.args.num_words_title)
                self.update_dict(self.news, docid, title_index)

    def process_title_index(self):
        news_num = len(self.news) + 1

        self.news_title = np.zeros((news_num, self.args.num_words_title), dtype='int32')
        self.news_title_attmask = np.zeros((news_num, self.args.num_words_title), dtype='int32')

        for key in self.news:
            title = self.news[key]
            doc_index = self.nid2index[key]

            self.news_title[doc_index] = title['input_ids']
            self.news_title_attmask[doc_index] = title['attention_mask']


