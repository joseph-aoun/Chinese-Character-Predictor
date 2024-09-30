from collections.abc import Sequence, Mapping
from typing import Type, Tuple
from collections import defaultdict
import math
import utils
# from tqdm import tqdm

NgramType: Type = Type["Ngram"]

class Ngram(object):

    def __init__(self: NgramType,
                 n: int,
                 data: Sequence[Sequence[str]]
                 ) -> None:
        self.n = n
        self.vocab: utils.Vocab = utils.Vocab()
        self.counts = [defaultdict(int) for _ in range(n + 1)]
        self.ngram_counts = defaultdict(int)
        self.continuations = [defaultdict(set) for _ in range(n + 1)]
        self.memo = {}

        def calc(i, line):
            for j in range(i, len(line) + 1):
                ngram = tuple(line[j - i:j])
                self.counts[i][ngram] += 1
                self.continuations[i-1][ngram[:-1]].add(ngram[-1])
                self.ngram_counts[ngram[:-1]] += 1

        for i in range(1, n + 1):
            for line in data:
                line = [utils.START_TOKEN] * (i - 1) + line + [utils.END_TOKEN]
                calc(i, line)
                for token in line:
                    self.vocab.add(token)
                    
        self.s = sum(self.counts[1].values())

    def caculate_prob(self, ngram, word, d=0.75):
        
        M = ngram + (word,)
        
        if M in self.memo:
            return self.memo[M]

        if len(ngram) == 0:
            self.memo[M] = (self.counts[len(ngram) + 1][ngram + (word,)] + 1) / (self.s + len(self.vocab) - 1)
            return self.memo[M]
        
        previous_prob = self.caculate_prob(ngram[1:], word, d)
        num = max(0, self.counts[len(M)][M] - d)
        denom = self.ngram_counts[ngram]
        
        if denom == 0:
            self.memo[M] = previous_prob
            return self.memo[M]

        self.memo[M] = num / denom + (d * len(self.continuations[len(ngram)][ngram])) / denom * previous_prob
        return self.memo[M]
    
    
    def start(self: NgramType) -> str:
        return (utils.START_TOKEN,) * (self.n - 1)

    def step(self: NgramType,
             q: tuple,
             w: tuple
             ) -> Tuple[str, Mapping[str, float]]:
        """Compute one step of the language model.

        Arguments:
        - q: The current state of the model
        - w: The most recently seen token (str)

        Return: (r, pb), where
        - r: The state of the model after reading `w`
        - pb: The log-probability distribution over the next token
        """
        res = tuple(list(q)[1:] + [w])
        prob_dist = defaultdict(float)
        sm = 0
        for x in self.vocab:
            if x != utils.START_TOKEN:
                prob_dist[x] = math.log(self.caculate_prob(res, x) + 1e-10)
                sm += math.exp(prob_dist[x])
        
        if sm < 1e-6:
            prob_dist[utils.UNK_TOKEN] = 0.0
            sm += 1
        
        return res, prob_dist