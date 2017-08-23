import os
import numpy as np
from PIL import Image

import nltk
from nltk.tokenize import RegexpTokenizer

import torch
import torch.utils.data as data
from torch.utils.serialization import load_lua
import torchvision.transforms as transforms


def split_sentence_into_words(sentence):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(sentence.lower())


class ReedICML2016(data.Dataset):
    def __init__(self, img_root, caption_root, classes_fllename,
                 word_embedding, max_word_length, img_transform=None):
        super(ReedICML2016, self).__init__()
        self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

        self.max_word_length = max_word_length
        self.img_transform = img_transform

        if self.img_transform == None:
            self.img_transform = transforms.ToTensor()

        self.data = self._load_dataset(img_root, caption_root, classes_fllename, word_embedding)

    def _load_dataset(self, img_root, caption_root, classes_filename, word_embedding):
        output = []
        with open(os.path.join(caption_root, classes_filename)) as f:
            lines = f.readlines()
            for line in lines:
                cls = line.replace('\n', '')
                filenames = os.listdir(os.path.join(caption_root, cls))
                for filename in filenames:
                    datum = load_lua(os.path.join(caption_root, cls, filename))
                    raw_desc = datum['char'].numpy()
                    desc, len_desc = self._get_word_vectors(raw_desc, word_embedding)
                    output.append({
                        'img': os.path.join(img_root, datum['img']),
                        'desc': desc,
                        'len_desc': len_desc
                    })
        return output

    def _get_word_vectors(self, desc, word_embedding):
        output = []
        len_desc = []
        for i in range(desc.shape[1]):
            words = self._nums2chars(desc[:, i])
            words = split_sentence_into_words(words)
            word_vecs = torch.Tensor([word_embedding[w] for w in words])
            # zero padding
            if len(words) < self.max_word_length:
                word_vecs = torch.cat((
                    word_vecs,
                    torch.zeros(self.max_word_length - len(words), word_vecs.size(1))
                ))
            output.append(word_vecs)
            len_desc.append(len(words))
        return torch.stack(output), len_desc

    def _nums2chars(self, nums):
        chars = ''
        for num in nums:
            chars += self.alphabet[num - 1]
        return chars

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        img = Image.open(datum['img'])
        img = self.img_transform(img)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        desc = datum['desc']
        len_desc = datum['len_desc']
        # randomly select one sentence
        selected = np.random.choice(desc.size(0))
        desc = desc[selected, ...]
        len_desc = len_desc[selected]
        return img, desc, len_desc
