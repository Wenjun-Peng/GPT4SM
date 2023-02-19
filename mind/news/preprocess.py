from six.moves.urllib.parse import urlparse
from collections import Counter
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re
import os

from ..utils import MODEL_CLASSES

def get_domain(url):
    domain = urlparse(url).netloc
    return domain

def filter_emoji(content):
    try:
        # Wide UCS-4 build
        cont = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+')
    except re.error:
        # Narrow UCS-2 build
        cont = re.compile(u'('u'\ud83c[\udf00-\udfff]|'u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'u'[\u2600-\u2B55])+')
    return cont.sub (u'', content)
           

class NewsInfo:
    def __init__(self, args, category_dict=None, subcategory_dict=None, domain_dict=None):
        self.args = args

        self.news = {}
        self.news_index = {}
        self.title_index = {}

        if self.args.mode == 'test':
            self.category_dict, self.subcategory_dict, self.domain_dict, \
                 = category_dict, subcategory_dict, domain_dict
        else:
            self.category_dict, self.subcategory_dict, self.domain_dict = {}, {}, {}

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        self.tokenizers = tokenizer_class.from_pretrained(os.path.join(self.args.bert_path, args.tokenizer_name),
                                               do_lower_case=True)

    def update_dict(self, dict, key, value=None):
        if key not in dict:
            if value is None:
                dict[key] = len(dict) + 1
            else:
                dict[key] = value
        return dict

    def sent_tokenize(self, sent, max_len):
        assert isinstance(sent, str)
        sent_split = self.tokenizers(sent, max_length=max_len, pad_to_max_length=True, truncation=True)
        return sent_split

    def _parse_news_attrs(self, attr_raw_values):
        parser = {
            'title': (self.sent_tokenize, [], {"max_len":self.args.num_words_title}),
            'body': (self.sent_tokenize, [], {"max_len": self.args.num_words_body}),
            'abstract': (self.sent_tokenize, [], {"max_len": self.args.num_words_abstract}),
            'category': (lambda x: x, None, {}),
            'subcategory': (lambda x: x, None, {}),
            'domain': (get_domain, None, {})
        }

        news_attrs = [
            self._parse_news_attr(
            attr_name, parser[attr_name], attr_raw_value
            ) for attr_name, attr_raw_value in 
            zip(
                ['title', 'abstract', 'body', 'category', 'domain', 'subcategory'],
                attr_raw_values
            )]

        return news_attrs

    def _parse_news_attr(self, attr_name, parser, attr_raw_value):
        parser_func, default_value, kwargs = parser
        if attr_name in self.args.news_attributes:
            return parser_func(attr_raw_value, **kwargs)
        else:
            return default_value

    def parse_line(self, line):
        doc_id, category, subcategory, title, abstract, url, _, _ = line.strip('\n').split('\t')
        body = ""
        title = " ".join([category, subcategory, title])
        self.update_dict(self.title_index, doc_id, title)
        title, abstract, body, category, domain, subcategory = self._parse_news_attrs(
            [title, abstract, body, category, url, subcategory]
        )
        self.update_dict(self.news, doc_id, [title, abstract, body, category, domain, subcategory])
        self.update_dict(self.news_index, doc_id)
        if self.args.mode == 'train':
            self.update_dict(self.category_dict, category)
            self.update_dict(self.subcategory_dict, subcategory)
            self.update_dict(self.domain_dict, domain)

    def process_news_file(self, file):
        with tf.io.gfile.GFile(file, "r") as f:
            for i, line in tqdm(enumerate(f)):
                self.parse_line(line)
            

def get_doc_input(news, news_index, category_dict, domain_dict,
                  subcategory_dict, args):
    news_num = len(news) + 1
    if 'title' in args.news_attributes:
        news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_attmask = np.zeros((news_num, args.num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_attmask = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((news_num, args.num_words_abstract),
                                 dtype='int32')
        news_abstract_attmask = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_attmask = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_attmask = np.zeros((news_num, args.num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_attmask = None

    if 'category' in args.news_attributes:
        news_category = np.zeros((news_num, 1), dtype='int32')
    else:
        news_category = None

    if 'domain' in args.news_attributes:
        news_domain = np.zeros((news_num, 1), dtype='int32')
    else:
        news_domain = None

    if 'subcategory' in args.news_attributes:
        news_subcategory = np.zeros((news_num, 1), dtype='int32')
    else:
        news_subcategory = None

    for key in tqdm(news):
        title, abstract, body, category, domain, subcategory = news[key]
        doc_index = news_index[key]

        if 'title' in args.news_attributes:
            news_title[doc_index] = title['input_ids']
            news_title_attmask[doc_index] = title['attention_mask']

        if 'abstract' in args.news_attributes:
            news_abstract[doc_index] = abstract['input_ids']
            news_abstract_attmask[doc_index] = abstract['attention_mask']

        if 'body' in args.news_attributes:
            news_body[doc_index] = body['input_ids']
            news_body_attmask[doc_index] = body['attention_mask']

        if 'category' in args.news_attributes:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        
        if 'subcategory' in args.news_attributes:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
        
        if 'domain' in args.news_attributes:
            news_domain[doc_index, 0] = domain_dict[domain] if domain in domain_dict else 0

    return news_title, news_title_attmask, news_abstract, news_abstract_attmask, \
           news_body, news_body_attmask, news_category, news_domain, news_subcategory

def get_doc_input_pop(news, news_index, category_dict, domain_dict,
                  subcategory_dict, args):
    news_num = len(news) + 1
    news_id = []
    news_id_title = {}
    if 'title' in args.news_attributes:
        news_title = np.zeros((news_num, args.num_words_title), dtype='int32')
        news_title_attmask = np.zeros((news_num, args.num_words_title), dtype='int32')
    else:
        news_title = None
        news_title_attmask = None

    if 'abstract' in args.news_attributes:
        news_abstract = np.zeros((news_num, args.num_words_abstract),
                                 dtype='int32')
        news_abstract_attmask = np.zeros((news_num, args.num_words_abstract), dtype='int32')
    else:
        news_abstract = None
        news_abstract_attmask = None

    if 'body' in args.news_attributes:
        news_body = np.zeros((news_num, args.num_words_body), dtype='int32')
        news_body_attmask = np.zeros((news_num, args.num_words_body), dtype='int32')
    else:
        news_body = None
        news_body_attmask = None

    if 'category' in args.news_attributes:
        news_category = np.zeros((news_num, 1), dtype='int32')
    else:
        news_category = None

    if 'domain' in args.news_attributes:
        news_domain = np.zeros((news_num, 1), dtype='int32')
    else:
        news_domain = None

    if 'subcategory' in args.news_attributes:
        news_subcategory = np.zeros((news_num, 1), dtype='int32')
    else:
        news_subcategory = None

    for key in tqdm(news):
        title, abstract, body, category, domain, subcategory = news[key]
        doc_index = news_index[key]
        news_id.append(key)
        news_id_title[key] = title

        if 'title' in args.news_attributes:
            news_title[doc_index] = title['input_ids']
            news_title_attmask[doc_index] = title['attention_mask']

        if 'abstract' in args.news_attributes:
            news_abstract[doc_index] = abstract['input_ids']
            news_abstract_attmask[doc_index] = abstract['attention_mask']

        if 'body' in args.news_attributes:
            news_body[doc_index] = body['input_ids']
            news_body_attmask[doc_index] = body['attention_mask']

        if 'category' in args.news_attributes:
            news_category[doc_index, 0] = category_dict[category] if category in category_dict else 0
        
        if 'subcategory' in args.news_attributes:
            news_subcategory[doc_index, 0] = subcategory_dict[subcategory] if subcategory in subcategory_dict else 0
        
        if 'domain' in args.news_attributes:
            news_domain[doc_index, 0] = domain_dict[domain] if domain in domain_dict else 0

    return news_title, news_title_attmask, news_abstract, news_abstract_attmask, \
           news_body, news_body_attmask, news_category, news_domain, news_subcategory, news_id, news_id_title
