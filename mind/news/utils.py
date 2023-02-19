
def sent_tokenize(tokenizers, sent, max_len):
    assert isinstance(sent, str)
    sent_split = tokenizers(sent, max_length=max_len, padding='max_length', truncation=True)
    return sent_split