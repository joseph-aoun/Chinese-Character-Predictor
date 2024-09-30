import unigram, data.charloader as charloader
def train_unigram() -> unigram.Unigram:
    data = charloader.load_chars_from_file("data/english/train")
    return unigram.Unigram(data)

def dev_unigram(unigram_model: unigram.Unigram) -> float:
    data = charloader.load_chars_from_file("data/english/dev")
    total_correct = total = 0
    mx = max(unigram_model.logprob.keys(), key=lambda x: unigram_model.logprob[x])
    for line in data:
        for chr in line:
            total += 1
            if chr == mx:
                total_correct += 1
    return (total_correct, total)

# unigram_model = train_unigram("data/english/train")
# x, y = dev_unigram(unigram_model, "hw1/data/english/dev")