from charpredictor import CharPredictor
def train_model():
    bigram = CharPredictor(2, "data/mandarin/charmap", "data/mandarin/train.han")
    return bigram

def split(line: str,
          delim: str = None
          ):
    line: str = line.rstrip('\r\n')
    if delim == '':
        return list(line)
    else:
        return line.split(delim)

def Read_parallel(ffilename: str,
                  efilename: str,
                  delim1: str = None,
                  delim2: str = None
                  ):
    """Read data from the files named by `ffilename` and `efilename`.

    The files should have the same number of lines.

    Arguments:
      - ffilename: str
      - efilename: str
      - delim: delimiter between symbols (default: any whitespace)
    Returns: list of pairs of lists of strings. <BOS> and <EOS> are added to all sentences.
    """
    data = []
    for (fline, eline) in zip(open(ffilename, encoding="utf-8"), open(efilename, encoding="utf-8")):
        fwords = split(fline, delim1)
        ewords = split(eline, delim2)
        data.append((fwords, ewords))
    return data

def dev_model(bigram):
    total = 0
    total_correct = 0
    dev_path = "data/mandarin/dev.pin"
    dev_correct_path = "data/mandarin/dev.han"
    
    for token, correct_token in Read_parallel(dev_path, dev_correct_path, ' ', ""):
        q = bigram.start()
        for a, b in zip(token, correct_token):
            total += 1
            prob = bigram.step(q, a)[1]
            mx = max(prob.keys(), key=lambda k: prob[k])
            if mx == b:
                total_correct += 1
            q = [b]
    return (total_correct, total)

def test_model(bigram):
    total = 0
    total_correct = 0
    test_path = "data/mandarin/test.pin"
    test_correct_path = "data/mandarin/test.han"
    
    for token, correct_token in Read_parallel(test_path, test_correct_path, ' ', ""):
        q = bigram.start()
        for a, b in zip(token, correct_token):
            total += 1
            prob = bigram.step(q, a)[1]
            mx = max(prob.keys(), key=lambda k: prob[k])
            if mx == b:
                total_correct += 1
            q = [b]
    return (total_correct, total)

# model = train_model()
# test = dev_model(model)
# print(test)
# test = test_model(model)
# print(test)