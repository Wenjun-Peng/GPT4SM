from datasets import Dataset, DatasetDict
from collections import defaultdict
import logging
from tqdm import tqdm

def convert_ads_tsv_dict(tsv_path, use_snippet):
    data_dict = defaultdict(list)
    label_count = defaultdict(int)
    with open(tsv_path) as f:
        for cnt, line in tqdm(enumerate(f)):
            label, _, query, keywords = line.strip().split('\t')
            
            label = int(label)
            if label >= 0:
                data_dict['label'].append(label)
                data_dict['idx'].append(cnt)
                data_dict['query'].append(query)
                data_dict['keywords'].append(keywords)
                label_count[int(label)] += 1
            # if cnt > 100 and 'train' in tsv_path:
            #     break
            # if cnt >= 320:
            #     break

    print(label_count)
    return data_dict


def convert_blue_tsv_dict(tsv_path, use_snippet):
    data_dict = defaultdict(list)
    # if 'train' in tsv_path:
    #     label_count = [0, 0, 0, 0, 0]
    # else:
    label_count = defaultdict(int)
    label_count2 = defaultdict(int)
    with open(tsv_path) as f:
        for cnt, line in tqdm(enumerate(f)):
            query, title, snippet, url, market, label, click_stream, achor, sentences, doc_lang = line.strip("\n").split("\t")
            # import pdb
            # pdb.set_trace()
            
            if 'train' in tsv_path:
                # label_count2[label] += 1
                if label == '0,0,0,0,0':
                    label = '1,0,0,0,0'
                label = [float(l) for l in label.split(',')]
                for i in range(len(label)):
                    if label[i] > 0:
                        data_dict['idx'].append(cnt)
                        data_dict['query'].append(query)
                        if use_snippet:
                            data_dict['title'].append(snippet)
                        else:
                            data_dict['title'].append(title)
                        data_dict['label'].append(i)
                        # label_count[i] += label[i]
                        label_count[i] += 1
            else:
                label = int(label)
                label_count[label] += 1
                data_dict['idx'].append(cnt)
                data_dict['query'].append(query)
                data_dict['title'].append(title)
                data_dict['label'].append(label)
            # if cnt > 100 and 'train' in tsv_path:
            #     break
            # if cnt >= 320:
            #     break
    print(label_count)
    # print(label_count2)
    return data_dict


CONVERT_FUNC = {
    'ads': convert_ads_tsv_dict,
    'blue': convert_blue_tsv_dict
}


def load_func(train_tsv_path, test_tsv_path, name, use_snippet=False):
    print("loading from: {}, {}".format(train_tsv_path, test_tsv_path))
    train_dict = CONVERT_FUNC[name](train_tsv_path, use_snippet)
    test_dict = CONVERT_FUNC[name](test_tsv_path, use_snippet)
    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)
    datasets = DatasetDict()
    datasets['train'] = train_dataset
    datasets['test'] = test_dataset

    return datasets


def check_blue(tsv_path):
    with open(tsv_path) as f:
        for cnt, line in tqdm(enumerate(f)):
            query, title, snippet, url, market, label, click_stream, achor, sentences, doc_lang = line.strip("\n").split("\t")

            label = [float(l) for l in label.split(',')]
            if len(label) != 5:
                print(label)
