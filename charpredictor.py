import data.charloader as charloader, ngram, math
from collections import defaultdict
from typing import Sequence

class CharPredictor(object):
    def __init__(self, n:int, charmap_path:str, training_path:str) -> None:
        self.n = n
        self.pronounciation_to_mandarin = defaultdict(lambda : set())
        self.mandarin_to_pronounciation = defaultdict(str)
        
        for line in charloader.load_lines_from_file(charmap_path):
            x, y = line.split()
            self.pronounciation_to_mandarin[y].add(x)
            self.mandarin_to_pronounciation[x] = y
            
        self.data = charloader.load_chars_from_file(training_path)
        self.ngram_model = ngram.Ngram(n, self.data)
        
    
    def candidates(self, token: str):
        s =  self.pronounciation_to_mandarin[token]
        
        if len(token) == 1:
            s.add(token)

        if token == "<space>":
            s.add(" ")

        return s

    def start(self) -> Sequence[str]:
        return self.ngram_model.start()
    
    def step(self, q: Sequence[str], w: str):
        prob = {c: self.ngram_model.caculate_prob(tuple(q), c) for c in self.candidates(w)}
        return [w], prob