import numpy as np

class GPTInfo:
    def __init__(self, emb_file, news_file):
        self.news_file = news_file
        self.emb_file = emb_file
        self.nid2index = {}

        self.record_size = 8 + 1536 * 4 * 2

    def process(self):
        self.process_news()
        self.process_emb()

    def update_dict(self, dict, key, value=None):
        if key not in dict:
            if value is None:
                dict[key] = len(dict) + 1
            else:
                dict[key] = value
        return dict

    def process_news(self):
        with open(self.news_file, 'r', encoding="utf8") as f:
            for line in f:
                line = line.strip('\n').split('\t')
                docid = int(line[0][1:])
                self.update_dict(self.nid2index, docid)
        

    def process_emb(self):
        self.title_embs = np.zeros((len(self.nid2index) + 1, 1536), dtype='float32')
        self.body_embs = np.zeros((len(self.nid2index) + 1, 1536), dtype='float32')

        with open(self.emb_file, 'rb') as f:
            while True:
                record = f.read(self.record_size)
                if not record:
                    break
                nid = int.from_bytes(record[:8], "big")
                title_emb = np.frombuffer(record[8: 8 + 1536 * 4], dtype="float32")
                body_emb = np.frombuffer(record[8 + 1536 * 4:], dtype="float32")

                nindex = self.nid2index[nid]
                self.title_embs[nindex] = title_emb
                self.body_embs[nindex] = body_emb


if __name__ == "__main__":
    news_info = NewsInfo(
        "/vc_data/users/v-jingweiyi/MIND/gpt_embedding/emb",
        "/vc_data/users/v-jingweiyi/MIND/all_news.tsv"
    )
    news_info.process()