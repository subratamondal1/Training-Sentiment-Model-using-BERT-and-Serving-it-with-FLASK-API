import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()

        self.bert = transformers.BertModel.from_pretrained(
            pretrained_model_name_or_path = config.BERT_PATH
        )
        self.bert_drop = nn.Dropout(p = 0.3)
        self.out_layer = nn.Linear(in_features=768, out_features=1)

    def forward(self, ids, mask, token_type_ids):
        out1, out2 = self.bert(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        bert_out = self.bert_drop(out2)
        output = self.out_layer(bert_out)
        return output


