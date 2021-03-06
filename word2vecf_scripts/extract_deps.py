# extract dependency pairs for word2vecf from either a conll file or a tree file
# modified to stanford dependencies instead of google universal-treebank annotation scheme.
# zcat treebank.gz |python extract_deps.py |gzip - > deps.gz

import sys
import re
from collections import defaultdict


def read_sent(fh, format):
    if format == 'conll':
        return read_conll(fh)
    elif format == 'stanford':
        return read_stanford(fh)
    else:
        raise LookupError('unknown sentence format %s' % format)


# conll format example:
# 1       during  _       IN      IN      _       7       prep
def read_conll(fh):
    root = (0, '*root*', -1, 'rroot')
    tokens = [root]
    for line in fh:
        if lower: line = line.lower()
        tok = line.strip().split('\t')
        if len(tok) != 7:
            # print('read_conll check : ', str(tok), file=sys.stderr)
            # tokens = []
            # break
            continue
        if not tok:
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            tokens.append((int(tok[0]), tok[1], int(tok[5]), tok[6]))
    if len(tokens) > 1:
        yield tokens


line_extractor = re.compile('([a-z]+)\(.+-(\d+), (.+)-(\d+)\)')


# stanford parser output example:
# num(Years-3, Five-1)
def read_stanford(fh):
    root = (0, '*root*', -1, 'rroot')
    tokens = [root]
    for line in fh:
        if lower: line = line.lower()
        tok = line_extractor.match(line)
        if not tok:
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            tokens.append((int(tok.group(4)), tok.group(3), int(tok.group(2)), tok.group(1)))
    if len(tokens) > 1:
        yield tokens


def read_vocab(fh):
    v = {}
    for line in fh:
        if lower: line = line
        line = line.strip().split()
        if len(line) != 2: continue
        if int(line[1]) >= THR:
            v[line[0]] = int(line[1])
        else:
            sys.stderr.write('read_vocab: less than THR, %s %s' % (line[0], line[1]))
    return v


if __name__ == '__main__':

    if len(sys.argv) < 3:
        sys.stderr.write(
            "Usage: parsed-file | %s <conll|stanford> <vocab-file> [<min-count>] > deps-file \n" % sys.argv[0])
        sys.exit(1)

    format = sys.argv[1]
    vocab_file = sys.argv[2]

    try:
        sys.stdin = open(sys.argv[4])
    except IndexError:
        print('using console input', file=sys.stderr)

    try:
        THR = int(sys.argv[3])
    except IndexError:
        THR = 100

    lower = True

    vocab = set(read_vocab(open(vocab_file)).keys())
    print("vocab:", len(vocab), file=sys.stderr)
    for i, sent in enumerate(read_sent(sys.stdin, format)):
        if i % 10000 == 0:
            print(i, file=sys.stderr)
        for tok in sent[1:]:
            par_ind = tok[2]
            par = sent[par_ind]
            m = tok[1]
            if m not in vocab: continue
            rel = tok[3]
            #      Universal dependencies
            #      if rel == 'adpmod': continue # this is the prep. we'll get there (or the PP is crappy)
            #      if rel == 'adpobj' and par[0] != 0:

            #     Stanford dependencies
            if rel == 'prep': continue  # this is the prep. we'll get there (or the PP is crappy)
            if rel == 'pobj' and par[0] != 0:

                ppar = sent[par[2]]
                rel = "%s:%s" % (par[3], par[1])
                h = ppar[1]
            else:
                h = par[1]
            if h not in vocab and h != '*root*': continue
            if h != '*root*': print(h, "_".join((rel, m)))
            print(m, "I_".join((rel, h)))
