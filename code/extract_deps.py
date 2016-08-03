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
        # if lower:
        line = line.lower()
        tok = line.strip().split('\t')

        if tok == ['']:
            if len(tokens) > 1:
                yield tokens
            tokens = [root]
        else:
            # if len(tok) == 7:
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
    for i, line in enumerate(fh):
        if lower:
            line = line.lower()

        line = line.strip().split()

        if len(line) != 2:
            # print('read_vocab: invalid format at line {}: {}'.format(i, line), file=sys.stderr)
            continue

        if int(line[1]) >= THR:
            v[line[0]] = int(line[1])
        else:
            print('read_vocab: less than THR, %s %s' % (line[0], line[1]), file=sys.stderr)
    return v


if __name__ == '__main__':

    if len(sys.argv) < 3:
        sys.stderr.write(
            "Usage: parsed-file | %s <conll|stanford> <vocab-file> [<min-count>] > deps-file \n" % sys.argv[0])
        sys.exit(1)

    format = sys.argv[1]
    vocab_file = sys.argv[2]

    try:
        THR = int(sys.argv[3])
    except IndexError:
        THR = 100

    lower = True

    print("format:", format, file=sys.stderr)
    print("vocab_file:", vocab_file, file=sys.stderr)
    print("THR:", THR, file=sys.stderr)

    try:
        sys.stdin = open(sys.argv[4])
    except IndexError:
        print('using console input', file=sys.stderr)

    try:
        sys.stdout = open(sys.argv[5], 'w')
    except IndexError:
        print('using console output', file=sys.stderr)

    vocab = set(read_vocab(open(vocab_file, encoding='iso-8859-1')).keys())
    print("vocab:", len(vocab), file=sys.stderr)

    for i, sent in enumerate(read_conll(sys.stdin)):
        if i % 100000 == 0:
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

    print('lines:', i, file=sys.stderr)
