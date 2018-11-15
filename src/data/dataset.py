import re,sys,os
import unicodedata
import numpy as np
from torch.utils.data.dataset import Dataset

# Transalation
class TrainDataset(Dataset):
    class Voc(object):
        def __init__(self, name):
            self.name = name
            self.word2index = {}
            self.word2count = {}
            self.index2word = {0: "SOS", 1: "EOS", 2:"PAD"}
            self.n_words = 3  # Count SOS and EOS

        def add_sentence(self, sentence):
            for word in sentence.split(' '):
                self.add_word(word)

        def add_word(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word[self.n_words] = word
                self.n_words += 1
            else:
                self.word2count[word] += 1
                
    def __init__(self,lang1,lang2,MAX_LENGTH = 10,reverse=False):
        self.MAX_LENGTH = MAX_LENGTH
        self.SOS_token = 0
        self.EOS_token = 1
        self.PAD_token = 2
        self.input_voc,self.output_voc,self.pairs = self.prepare_data(lang1,lang2,reverse)
        input_data = list(map(lambda x:self.indexes_from_sentence(self.input_voc,x[0])+[self.EOS_token],self.pairs))
        output_data = list(map(lambda x:self.indexes_from_sentence(self.input_voc,x[0])+[self.EOS_token],self.pairs))
        self.input_lengths = np.array([len(seq) for seq in input_data])
        self.output_lengths = np.array([len(seq) for seq in output_data])
        self.input_data = self.zeroPadding(input_data,self.PAD_token)
        self.output_data = self.zeroPadding(output_data,self.PAD_token)
        
    def filter_pair(self, p):
        return len(p[0].split(' ')) < self.MAX_LENGTH and len(p[1].split(' ')) < self.MAX_LENGTH

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self.filter_pair(pair)]

    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalize_string(self, s):
        s = self.unicode_to_ascii(s.strip()).lower().strip()
        s = re.sub("([.!?])", " \1", s)
        s = re.sub("[^a-zA-Z.!?s]+", " ", s)
        return s.strip()

    def read_lang(self, lang1, lang2, reverse=False):
        print("Reading lines...")
        # combine every two lines into pairs and normalize
        with open(os.path.join(sys.path[-1],'data/interim/%s-%s.txt' % (lang1, lang2)), encoding='utf-8') as f:
            content = f.readlines()
        lines = [x.strip() for x in content]
        pairs = [[self.normalize_string(s) for s in line.split('\t')] for line in lines]
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_voc = self.Voc(lang2)
            output_voc = self.Voc(lang1)
        else:
            input_voc = self.Voc(lang1)
            output_voc = self.Voc(lang2)
        return input_voc, output_voc, pairs

    def indexes_from_sentence(self, voc, sentence):
        return [voc.word2index[word] for word in sentence.split(' ')]
    
    # batch_first: true -> false, i.e. shape: seq_len * batch
    def zeroPadding(self,data, fillvalue):
        pad = len(max(data, key=len))
        return np.array([i + [fillvalue]*(pad-len(i)) for i in data])

    def prepare_data(self, lang1, lang2, reverse=False):
        input_voc, output_voc, pairs = self.read_lang(lang1, lang2, reverse)
        print("Read %s sentence pairs" % len(pairs))
        pairs = self.filter_pairs(pairs)
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_voc.add_sentence(pair[0])
            output_voc.add_sentence(pair[1])
        print("Counted words:")
        print(input_voc.name, input_voc.n_words)
        print(output_voc.name, output_voc.n_words)
        return input_voc,output_voc,pairs
        
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.input_data[idx],self.input_lengths[idx],self.output_data[idx],self.output_lengths[idx]
