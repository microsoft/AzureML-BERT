from tqdm import tqdm
from typing import Tuple
from random import shuffle
import pickle
import random

from pytorch_pretrained_bert.tokenization import BertTokenizer


def truncate_input_sequence(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


class TokenInstance:
    def __init__(self, tokens_a, tokens_b, is_next):
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next  # 0 is if in continuation, 1 if is random

    def get_values(self):
        return (self.tokens_a, self.tokens_b, self.is_next)


class PretrainingDataCreator:
    def __init__(self, path, tokenizer: BertTokenizer,  max_seq_length, readin: int = 2000000, dupe_factor: int = 5, small_seq_prob: float = 0.1):
        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob

        documents = []
        instances = []
        with open(path, encoding='utf-8') as fd:
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                # Expected format (Q,T,U,S,D)
                # query, title, url, snippet, document = line.split('\t')
                # ! remove this following line later
                document = line
                if len(document.split("<sep>")) <= 3:
                    continue
                lines = document.split("<sep>")
                document = []
                for seq in lines:
                    document.append(tokenizer.tokenize(seq))
                # document = list(map(tokenizer.tokenize, lines))
                documents.append(document)

        documents = [x for x in documents if x]

        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)
        self.documents = None
        documents = None

    def __len__(self):
        return self.len

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def save(self, filename):
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    @staticmethod
    def load(filename):
        print("Loading filename {}".format(filename))
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def create_training_instance(self, index):
        document = self.documents[index]

        # Need to add [CLS] + 2*[SEP] tokens
        max_num_tokens = self.max_seq_length - 3

        # We want to maximize the inp sequence but also want inputs similar
        # to our generic task inputs which will be compartively smaller
        # than the data on which we intend to pre-train.
        target_seq_length = max_num_tokens
        if random.random() < self.small_seq_prob:
            target_seq_length = random.randint(5, max_num_tokens)

        # Need to make the sequences split for NSP task for interesting
        # rather than choosing some arbitrary point. If not the NSP
        # task might become way too easy.
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document)-1 or current_length >= target_seq_length:
                if current_chunk:
                    # `a_end` is how many segments from `current_chunk` go into the `A`
                    # (first) sentence.
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []

                    # Random Next
                    is_random_next = False
                    if len(current_chunk) == 1 or random.random() < 0.5:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # Pick a random document
                        for _ in range(10):
                            random_doc_index = random.randint(
                                0, len(self.documents) - 1)
                            if random_doc_index != index:
                                break

                        random_doc = self.documents[random_doc_index]
                        random_start = random.randint(0, len(random_doc)-1)
                        for j in range(random_start, len(random_doc)):
                            tokens_b.extend(random_doc[j])
                            if len(tokens_b) >= target_b_length:
                                break

                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments

                    # Actual Next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])

                    truncate_input_sequence(tokens_a, tokens_b, max_num_tokens)

                    assert len(tokens_a) >= 1
                    assert len(tokens_b) >= 1

                    instances.append(TokenInstance(
                        tokens_a, tokens_b, int(is_random_next)))

                current_chunk = []
                current_length = 0
            i += 1

        return instances


class WikiNBookCorpusPretrainingDataCreator(PretrainingDataCreator):
    def __init__(self, path, tokenizer: BertTokenizer,  max_seq_length: int = 512, readin: int = 2000000, dupe_factor: int = 6, small_seq_prob: float = 0.1):
        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob

        documents = []
        instances = []
        with open(path, encoding='utf-8') as fd:
            document = []
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                # document = line
                # if len(document.split("<sep>")) <= 3:
                #     continue
                if len(line) == 0:  # This is end of document
                    documents.append(document)
                    document = []
                if len(line.split(' ')) > 2:
                    document.append(tokenizer.tokenize(line))
            if len(document) > 0:
                documents.append(document)

        documents = [x for x in documents if x]
        print(documents[0])
        print(len(documents))
        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)
        self.documents = None
        documents = None

class WikiPretrainingDataCreator(PretrainingDataCreator):
    def __init__(self, path, tokenizer: BertTokenizer,  max_seq_length: int = 512, readin: int = 2000000, dupe_factor: int = 6, small_seq_prob: float = 0.1):
        self.dupe_factor = dupe_factor
        self.max_seq_length = max_seq_length
        self.small_seq_prob = small_seq_prob

        documents = []
        instances = []
        with open(path, encoding='utf-8') as fd:
            document = []
            for i, line in enumerate(tqdm(fd)):
                line = line.replace('\n', '')
                # document = line
                # if len(document.split("<sep>")) <= 3:
                #     continue
                if len(line) > 0 and line[:2] ==  "[[" : # This is end of document
                    documents.append(document)
                    document = []
                if len(line.split(' ')) > 2:
                    document.append(tokenizer.tokenize(line))
            if len(document) > 0:
                documents.append(document)

        documents = [x for x in documents if x]
        self.documents = documents
        for _ in range(self.dupe_factor):
            for index in range(len(self.documents)):
                instances.extend(self.create_training_instance(index))

        shuffle(instances)
        self.instances = instances
        self.len = len(self.instances)
        self.documents = None
        documents = None