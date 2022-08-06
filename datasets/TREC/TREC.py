import json
import datasets
import os

class TREC(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'text': datasets.Value('string'),
                'labels': datasets.features.ClassLabel(names=['description', 'entity', 'abbreviation', 'human', 'numeric', 'location'])
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split(f'{split}_{seed}'), gen_kwargs={"filepath": f'./datasets/TREC/{seed}/{split}.tsv'})
            for seed in [8, 13, 42, 50, 60] for split in ['train', 'dev']
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                text, labels = line.strip().split('\t')
                yield id_, {'text': text, 'labels': labels}