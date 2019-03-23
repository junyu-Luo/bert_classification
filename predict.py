from tqdm import tqdm, trange
import tensorflow as tf
import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tokenization_word as tokenization
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root_path = r'./'
bert_path = './chinese_L-12_H-768_A-12'

flags = tf.flags

flags.DEFINE_string("data_dir", os.path.join(root_path, 'data'), "The input datadir.", )

flags.DEFINE_string("bert_config_file", os.path.join(bert_path, 'bert_config.json'),
                    "The config json file corresponding to the pre-trained BERT model.")

flags.DEFINE_string("test_file", 'test.xlsx', "The name of the task to train.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("output_dir", os.path.join(root_path, 'model'),
                    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("init_checkpoint", os.path.join(bert_path, 'pytorch_model.bin'),
                    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text.")

flags.DEFINE_integer("max_seq_length", 48, "The maximum total input sequence length after WordPiece tokenization.")

# flags.DEFINE_boolean('clean', True, 'remove the files which created by last training')

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("no_cuda", False, "Whether not to use CUDA when available")

flags.DEFINE_integer("train_batch_size", 256, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 256, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("save_model_epoch", 5, "save model ")

flags.DEFINE_float("num_train_epochs", 15.0, "Total number of training epochs to perform.")

flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float("warmup_proportion", 0.1,
                   "Proportion of training to perform linear learning rate warmup for. ""E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 50, "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 50, "How many steps to make in each estimator call.")

flags.DEFINE_integer("local_rank", -1, "local_rank for distributed training on gpus")

flags.DEFINE_integer("seed", 1, "random seed for initialization")

flags.DEFINE_integer("gradient_accumulation_steps", 1,
                     "Number of updates steps to accumualte before performing a backward/update pass.")

flags.DEFINE_bool("optimize_on_cpu", False,
                  "Whether to perform optimization and keep the optimizer averages on CPU")

flags.DEFINE_bool("fp16", True, "Whether to use 16-bit float precision instead of 32-bit")

flags.DEFINE_float('loss_scale', 128.0, 'Loss scaling, positive power of 2 values can improve fp16 convergence.')

args = flags.FLAGS


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def clean_data(string):
	try:
	    if type(string) == np.float64:
	        string = ''
	    if type(string) == int:
	        string = str(string)
	    if type(string) != float:
	        string = string.replace("　","").replace(" ","").replace("\n","").replace("\r","").replace("\t","").replace("&","").replace("#","").replace("@","").replace("▲","").replace("◤","")#.strip()
	    else:
	        string = ''
	    return string
	except:
		string = '未识别出特殊字符'
		return string
    

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()



    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        file_in = open(input_file, "rb")
        lines = []
        for line in file_in:
            lines.append(line.decode("utf-8").split("    "))  # .decode("gbk")
        return lines



    @classmethod
    def _read_excel(self, input_file):
        """Reads a tab separated value file."""
        result = []
        NAME = []
        SPECS = []
        USE_NAME = []
        # CODE = []
        data = pd.read_excel(input_file) # './data/train.xlsx'
        for i in trange(len(data)):
            NAME.append(clean_data(data['NAME'][i]))
            SPECS.append(clean_data(data['SPECS'][i]))
            USE_NAME.append(clean_data(data['USE_NAME'][i]))
            # CODE.append(cls.clean_data(data['CODE'][i]))
        assert len(NAME) == len(SPECS) == len(USE_NAME)
        for i in trange(len(NAME)):
            result_str = '无' + '    ' + NAME[i] + '。' + USE_NAME[i] + '。' + SPECS[i] + '。'
            result.append(result_str.split('    '))
        return result
        # print(len(NAME),NAME)
        # print(len(SPECS),SPECS)
        # print(len(USE_NAME),USE_NAME)
        # print(len(CODE),CODE)
        # for i in trange(len(NAME)):
        #     write_file('train.tsv', CODE[i] + '    ' + NAME[i] + '。' + SPECS[i] + '。' + USE_NAME[i] + '。')


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class NewsProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def __init__(self):
        self.labels = set()

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_predict_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_excel(os.path.join(data_dir, args.test_file)), "test")

    def get_labels(self):
        """See base class."""
        return list(self.labels)

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if len(line) == 2:
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            self.labels.add(label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # label_id = label_map[example.label]
        label_list = load_pkl('label_list.pkl')
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[label] = i
        try:
            label_id = label_map[example.label]
        except:
            label_id = 0

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def save_pkl(filename, content):
    file = open('./pkl/' + filename, 'wb')
    pickle.dump(content, file)
    file.close()


def load_pkl(filename):
    with open('./pkl/' + filename, 'rb') as file:
        return pickle.load(file)


processor = NewsProcessor()
predict_examples = processor.get_predict_examples(args.data_dir)

label_list = load_pkl('label_list.pkl')
tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
eval_features = convert_examples_to_features(predict_examples, label_list, args.max_seq_length, tokenizer)

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
# model.load_state_dict(torch.load('./model/model_1.pkl', map_location='cpu'))
# model = torch.load('./model/model_5_0.9772.pkl')
model = torch.load('./model/model_9_0.9353.pkl') #model_14_0.9784.pkl
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_predict_examples = 0, 0
tem_lis = []
confidence_lis = []
difference_lis = []
decision = []
pred_out = pd.read_excel(os.path.join(args.data_dir, args.test_file))
NAME = []
SPECS = []
USE_NAME = []
for i in trange(len(pred_out)):
    NAME.append(clean_data(pred_out['NAME'][i]))
    SPECS.append(clean_data(pred_out['SPECS'][i]))
    USE_NAME.append(clean_data(pred_out['USE_NAME'][i]))
assert len(NAME) == len(SPECS) == len(USE_NAME)
for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)
    with torch.no_grad():
        tmp_eval_loss, pred_array = model(input_ids, segment_ids, input_mask, label_ids)
        # print(tmp_eval_loss.item())
        pred_array = pred_array.detach().cpu().numpy()
        for i in range(pred_array.shape[0]):
            pred_lis = list(pred_array[i])
            big = max(pred_lis)
            confidence_lis.append(big)
            tem_lis.append(label_list[pred_lis.index(max(pred_lis))])
            pred_lis.remove(max(pred_lis))
            difference_lis.append(big - max(pred_lis))
            decision.append(big * (big - max(pred_lis)))

code2name = load_pkl('code2name.pkl')
code2use_name = load_pkl('code2use_name.pkl')
shangpingming = []
geiyaotujing = []
for i in trange(len(tem_lis)):
    shangpingming.append(code2name[tem_lis[i]])
    geiyaotujing.append(code2use_name[tem_lis[i]])

df = pd.DataFrame({'NAME':NAME,'SPECS':SPECS,'USE_NAME':USE_NAME,'pred':tem_lis,'商品名':shangpingming,'给药途径':geiyaotujing,'confidence':confidence_lis,'difference':difference_lis,'decision':decision})
df.to_excel("pred_test_out.xlsx")

# print(len(tem_lis))
# for i in trange(len(tem_lis)):
#     pred_out['pred'][i] = tem_lis[i]
# pred_out.to_excel("pred_test.xlsx")



        # print(pred_array.shape[0])
        # pred_lis = list(pred_array[0])
        # print(label_list[pred_lis.index(max(pred_lis))])

