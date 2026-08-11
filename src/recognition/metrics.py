import editdistance

def normalize(s):
    return ''.join(c for c in s.lower() if c.isalnum())

def word_accuracy(preds, labels):
    correct = sum(normalize(p) == normalize(l) for p, l in zip(preds, labels))
    return correct / len(labels)

def cer(preds, labels):
    total_dist = sum(editdistance.eval(normalize(p), normalize(l)) for p, l in zip(preds, labels))
    total_chars = sum(len(normalize(l)) for l in labels)
    return total_dist / total_chars