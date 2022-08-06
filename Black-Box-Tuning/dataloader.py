import datasets
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
from transformers import RobertaTokenizer


# 从huggingface datasets脚本中读取数据
def load_hf_dataset(task_name: str = 'SST-2', seed: int = 42, split: str = 'train') -> datasets.Dataset:
    """
    Please choose from:
    :param task_name: 'AGNews', 'MRPC', 'SNLI', 'SST-2', 'TREC', 'Yelp'
    :param seed: 8, 13, 42, 50, 60
    :param split: 'train', 'dev'
    """
    dataset = datasets.load_dataset(
        path=f'./dataset/{task_name}/{task_name}.py',
        split=f'{split}_{seed}'
    )
    return dataset



def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
    mask_pos = []
    for input_ids in input_encodings['input_ids']:
        mask_pos.append(input_ids.index(tokenizer.mask_token_id))
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'mask_pos': mask_pos,
        'labels': target_encodings['input_ids'],
    }

    return encodings


class SST2Loader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s . It was %s .' % (prompt, example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['text'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='SST-2', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class YelpPLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s .' % (prompt, example['text'].replace("\\n", " "))
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s .' % (example['text'].replace("\\n", " "))
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='Yelp', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])

        def convert_to_features_yelp(example_batch, tokenizer):
            input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], max_length=509, truncation=True)
            target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
            template = tokenizer.encode(f'It was {tokenizer.mask_token}', add_special_tokens=False)
            input_encodings['input_ids'] = [ids[:-1] + template + [tokenizer.sep_token_id] for ids in input_encodings['input_ids']]
            input_encodings['attention_mask'] = [am + [1] * 3 for am in input_encodings['attention_mask']]
            mask_pos = []
            for input_ids in input_encodings['input_ids']:
                mask_pos.append(input_ids.index(tokenizer.mask_token_id))
            encodings = {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'mask_pos': mask_pos,
                'labels': target_encodings['input_ids'],
            }

            return encodings

        dataset = dataset.map(partial(convert_to_features_yelp, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class AGNewsLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Tech"
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s News: %s' % (prompt, self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s News: %s' % (self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='AGNews', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MRPCLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='MRPC', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle



class SNLILoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "Maybe",
            2: "No",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['text1'], self.tokenizer.mask_token, example['text2'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='SNLI', split=split, seed=seed)
        dataset = dataset.filter(lambda example: example['labels'] in [0, 1, 2])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class TRECLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "description",
            1: "entity",
            2: "abbreviation",
            3: "human",
            4: "numeric",
            5: "location"
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s question: %s ' % (prompt, self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['labels']]
        else:
            example['input_text'] = '%s . Topic: %s' % (self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['labels']]
        return example

    def _load(self, split, seed) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = load_hf_dataset(task_name='TREC', split=split, seed=seed)
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits, seed) -> DataBundle:
        datasets = {name: self._load(name, seed) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle