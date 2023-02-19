import random

from torch.utils.data import Dataset

def news_sample(news, ratio):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


class BehaviorTrainDataset(Dataset):
    def __init__(self, args, behavior_file, nid2index):
        self.args = args
        self.nid2index = nid2index

        with open(behavior_file, 'r') as f:
            self.lines = f.readlines()
        
        self.process_file()

    def process_file(self):
        self.samples = []
        for line_idx, line in enumerate(self.lines):
            cand_news = line.strip("\n").split("\t")[4].split(' ')
            poss_news = [i for i in cand_news if i.split('-')[1] == '1']
            for poss_idx in range(len(poss_news)):
                self.samples.append([line_idx, poss_idx])

    def __len__(self):
        return len(self.samples)

    def trans_to_nindex(self, nids):
        return [self.nid2index[int(i[1:])] if int(i[1:]) in self.nid2index else 0 for i in nids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length - len(x)) + x[-fix_length:]
            mask = [0] * (fix_length - len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[:fix_length] + [padding_value] * (fix_length - len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (len(x) - fix_length)
        return pad_x, mask

    def __getitem__(self, idx):
        line_idx, poss_idx = self.samples[idx]
        line = self.lines[line_idx]

        line = line.strip("\n").split("\t")
        click_docs = line[3].split()

        click_docs, log_mask = self.pad_to_fix_len(
            self.trans_to_nindex(click_docs), self.args.user_log_length
        )

        sess_news = [i.split("-") for i in line[4].split()]

        poss = [i[0] for i in sess_news if i[-1] == "1"][poss_idx]
        sess_neg = [i[0] for i in sess_news if i[-1] == "0"]

        poss = self.trans_to_nindex([poss])
        sess_neg = self.trans_to_nindex(sess_neg)

        if len(sess_neg) > 0:
            neg_index = news_sample(list(range(len(sess_neg))), self.args.npratio)
            sam_negs = [sess_neg[i] for i in neg_index]
        else:
            sam_negs = [0] * self.npratio

        sample_news = poss + sam_negs

        return sample_news, click_docs, log_mask, 0


class BehaviorTestDataset(BehaviorTrainDataset):
    def __init__(self, args, behavior_file, nid2index):
        self.args = args
        self.nid2index = nid2index
        with open(behavior_file, 'r') as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]

        line = line.strip("\n").split("\t")
        click_docs = line[3].split()

        click_docs, log_mask = self.pad_to_fix_len(
            self.trans_to_nindex(click_docs), self.args.user_log_length
        )

        sess_news = [i.split("-") for i in line[4].split()]
        sess_label = [int(i[-1]) for i in sess_news]
        sess_news_ids = [i[0] for i in sess_news]

        sess_news = self.trans_to_nindex(sess_news_ids)

        return sess_news, click_docs, log_mask, sess_label
