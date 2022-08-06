import json
import datasets
import os

class MRPC(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'text1': datasets.Value('string'),
                'text2': datasets.Value('string'),
                'labels': datasets.features.ClassLabel(names=['NotEquivalent', 'Equivalent'])
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split(f'{split}_{seed}'), gen_kwargs={"filepath": f'./datasets/MRPC/{seed}/{split}.tsv'})
            for seed in [8, 13, 42, 50, 60] for split in ['train', 'dev']
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            for id_, line in enumerate(f.readlines()):
                text1, text2, labels = line.strip().split('\t')
                yield id_, {'text1': text1, 'text2': text2, 'labels': labels}