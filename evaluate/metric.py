import pickle
import xml.etree.ElementTree as ET
import re
import os
import numpy as np


def cosine(u,v):
    return 1 - (np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v))))

os.chdir('C:\\Users\\cop15rj\\IdeaProjects\\rishav-msc-project')

# read the embeddings

VEC_FILE = 'data/vecs.npy'
VOCAB_FILE = 'data/vecs.vocab'

vecs = np.load(VEC_FILE)
vocab = open(VOCAB_FILE).read().split()
word2vec = {w:i for i,w in enumerate(vocab)}

WVEC_FILE = 'data/contexts.npy'
CONTEXT_FILE = 'data/contexts.vocab'

cvecs = np.load(WVEC_FILE)
context = open(CONTEXT_FILE).read().split()
context2vec = {w:i for i, w in enumerate(context)}


def create_tokens(fh):
    root = (0, '*root*', -1, 'rroot')
    tokens = [root]
    for line in fh:
        line = line.lower()
        tok = line.strip().split('\t')
        if len(tok) != 7:
            continue
        if not tok:
            if len(tokens) > 1: return tokens
            tokens = [root]
        else:
            tokens.append((int(tok[0]), tok[1], int(tok[5]), tok[6]))
    return tokens


def calculate_dep(sent):
    dep = {}
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
        if h != '*root*':
            if h in dep:
                dep[h].append("_".join((rel, m)))
            else:
                dep[h] = []
                dep[h].append("_".join((rel, m)))

        if m in dep:
            dep[m].append("I_".join((rel, h)))
        else:
            dep[m] = []
            dep[m].append("I_".join((rel, h)))
    return dep

TARGETS_FILE = 'evaluate/data/targets.txt'
SUBSTITUTES_FILE = 'evaluate/data/substitutes.data'
INPUT_CONLL_FILE = 'evaluate/data/sentences.txt.conll'
# INPUT_FILE = 'evaluate/taskdata/trial/lexsub_trial.xml'

substitutes = pickle.load(open(SUBSTITUTES_FILE, 'rb'))
# print('substitutes',':',substitutes)

conllFile = open(INPUT_CONLL_FILE)

targets = open(TARGETS_FILE).read().split('\n')

Id = 0
for line in conllFile:
    sentence = []
    while line and line != '\n':
        sentence.append(line)
        line = conllFile.readline()

    sentenceDep = calculate_dep(create_tokens(sentence))

    target = targets[Id].split('.')[0]

    if target in word2vec:
        wIdx = word2vec[target]
        print('wIdx', ':', wIdx)

        subs = substitutes[target]
        print('subs', ':', subs)

        for iSub in subs:
            if iSub in word2vec:
                sIdx = word2vec[iSub]
                print('sIdx', ':', sIdx)

                cos = cosine(vecs[wIdx], vecs[sIdx])
                print('cosine({}, {}) = {}'.format(target, iSub, cos))

                for c in sentenceDep[target]:
                    print(c)

                    if c in context2vec:
                        cIdx = context2vec[c]
                        print('cIdx', ':', cIdx)
                    else:
                        print('context not in vocab', ':', c)

                break
            else:
                print('check - substitute word not in vocab: {}'.format(iSub))
    else:
        print('check - target word not in vocab: {}'.format(target))

    break
