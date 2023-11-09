import config
import torch


class BERTDataset:
    def __init__(self, review, target):
        self.review = review # list of reviews (text)
        self.target = target # list of target values (1s, 0s)
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX__LEN

    def __len__(self):
        return len(self.review) # total length of the dataset
    
    def __getitem__(self, item):
        review = str(self.review[item])
        review = " ".join(review.split()) # remove all the weird spaces

        # encode_plus encodes two strings at a time
        inputs = self.tokenizer.encode_plus(
            review, # first string
            None,    # second string
            add_special_tokens = True,
            max_length = self.max_len
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"] 
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)

        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] *  padding_length)

        return {
            "ids" : torch.tensor(ids, dtype = torch.long),
            "mask" : torch.tensor(mask, dtype = torch.long),
            "token_type_ids" : torch.tensor(token_type_ids, dtype = torch.long),
            "targets" : torch.tensor(self.target[item], dtype = torch.float)
        }
    
