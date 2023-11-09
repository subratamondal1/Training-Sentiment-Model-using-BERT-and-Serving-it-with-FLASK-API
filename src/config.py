# Will contain the configurations that I will need the most
import transformers

MAX__LEN = 512
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE =  4
EPOCHS = 10
TRAIN_NUM_WORKERS = 4
VALID_NUM_WORKERS = 1
BERT_PATH = "input/bert_base_uncased/"
MODEL_PATH = "model.bin"
TRAINING_FILE =  "input/imdb.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    pretrained_model_name_or_path = BERT_PATH, 
    do_lower_case = True
)