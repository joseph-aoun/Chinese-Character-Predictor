import ngram, data.charloader as charloader
import utils


def train_ngram(n: int = 5) -> ngram.Ngram:
    data = charloader.load_chars_from_file("data/english/train")
    return ngram.Ngram(n, data)


def dev_ngram(ngram_model: ngram.Ngram) -> float:
    data = charloader.load_chars_from_file("data/english/dev")
    total_correct = total = 0
    for line in data:
        q = ngram_model.start()
        for c_input, c_actual in zip([utils.START_TOKEN] + line, line + [utils.END_TOKEN]):
            q, p = ngram_model.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])
            total += 1
            total_correct += int(c_predicted == c_actual)

    return (total_correct, total)

def test_ngram(ngram_model: ngram.Ngram) -> float:
    data = charloader.load_chars_from_file("data/english/test")
    total_correct = total = 0
    # ngram_model = train_ngram(8)
    for line in data:
        q = ngram_model.start()
        for c_input, c_actual in zip([utils.START_TOKEN] + line, line + [utils.END_TOKEN]):
            q, p = ngram_model.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])
            total += 1
            total_correct += int(c_predicted == c_actual)
    return (total_correct, total)
