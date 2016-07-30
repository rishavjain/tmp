import pickle
import xml.etree.ElementTree as ET
import re
import os
import numpy as np
from scipy.spatial.distance import cosine
from extract_deps import read_conll

# read the embeddings

WVEC_FILE = '../data/2/vecs.npy'
WVOCAB_FILE = '../data/2/vecs.vocab'

wvecs = np.load(WVEC_FILE)
wvocab = open(WVOCAB_FILE).read().split()
w2vec = {w: i for i, w in enumerate(wvocab)}

CVEC_FILE = '../data/2/contexts.npy'
CONTEXT_FILE = '../data/2/contexts.vocab'

cvecs = np.load(CVEC_FILE)
cvocab = open(CONTEXT_FILE).read().split()
c2vec = {w: i for i, w in enumerate(cvocab)}


def calculate_contexts(tokens, targetIdx):
    """
    calculate dependencies
    """
    dep = []
    for i, _tokens in enumerate(tokens):
        for token in _tokens[1:]:
            parIdx = token[2]
            par = _tokens[parIdx]

            w = token[1]
            wIdx = token[0]

            if w not in wvocab:
                continue

            rel = token[3]

            if rel == 'prep':
                continue  # this is the prep. we'll get there (or the PP is crappy)

            if rel == 'pobj' and par[0] != 0:
                ppar = _tokens[par[2]]
                rel = "%s:%s" % (par[3], par[1])
                h = ppar[1]
                hIdx = ppar[0]
            else:
                h = par[1]
                hIdx = par[0]

            if h not in wvocab and h != '*root*':
                continue

            if h != '*root*' and hIdx == targetIdx:
                dep.append("_".join((rel, w)))

            if wIdx == targetIdx:
                dep.append("I_".join((rel, h)))

    return dep


TARGETS_FILE = '../evaluate/data/targets.txt'
# SUBSTITUTES_FILE = '../evaluate/data/substitutes.data'
SUBSTITUTES_FILE = '../data/2/wv'


INPUT_CONLL_FILE = '../evaluate/data/sentences.txt.conll'
# INPUT_FILE = 'evaluate/taskdata/trial/lexsub_trial.xml'

OUTPUT_FILE_BEST = '../data/2/best.txt'
OUTPUT_FILE_OOT = '../data/2/oot.txt'

# subsDict = pickle.load(open(SUBSTITUTES_FILE, 'rb'))
subsDict = [line.strip().split()[0] for line in open(SUBSTITUTES_FILE, 'r').readlines()]
# print('substitutes',':',substitutes)

conllFile = open(INPUT_CONLL_FILE)

targetsFile = open(TARGETS_FILE)

Id = 0

bestFile = open(OUTPUT_FILE_BEST, 'w')
ootFile = open(OUTPUT_FILE_OOT, 'w')

def targetidx_sentence(conll, target, pos):
    if pos == 'a':
        pos = 'jj'

    for line in conll:
        line = line.lower().strip().split('\t')

        if target in line[1] and pos in line[3]:
            return int(line[0])

    raise RuntimeError('target not found in conll: {}'.format(target))

numsentences = 0

numdump = []
iD = 0
for line in conllFile:
    sentence = []
    while line and line != '\n':
        sentence.append(line)
        line = conllFile.readline()

    numsentences += 1

    # sentence <- conll data for one sentence in input
    sentenceTokens = read_conll(sentence)

    targetLine = targetsFile.readline().split()

    if not targetLine:
        continue

    lexelt = targetLine[0]
    target = targetLine[1]
    pos = lexelt.split('.')[1]

    if target not in w2vec:
        # # print('target word not in vocab: {}'.format(target))
        #
        # if lexelt.split('.')[0] in w2vec:
        #     # print('using the target word: {} instead of {}'.format(lexelt.split('.')[0], target))
        #     target = lexelt.split('.')[0]
        # else:
        continue

    wIdx = w2vec[target]

    tIdx = int(targetLine[2])
    contexts = calculate_contexts(sentenceTokens, tIdx)

    # print(contexts)

    # nSub = len(subsDict[lexelt.split('.')[0]])
    similarity = []
    # for i, sub in enumerate(subsDict[lexelt.split('.')[0]]):
    for i, sub in enumerate(subsDict):
        if sub in w2vec:
            sIdx = w2vec[sub]

            cos = cosine(wvecs[wIdx], wvecs[sIdx])
            # print('cosine({}, {}) = {}'.format(target, sub, cos))

            numcontexts = 0
            for context in contexts:
                if context in c2vec:
                    cIdx = c2vec[context]
                    cos += cosine(wvecs[sIdx], cvecs[cIdx])
                numcontexts += 1

            similarity.append((i, cos/(numcontexts + 1.0)))
        # else:
            # print('substitute word not in vocab: {}'.format(sub))

    # print(similarity)
    similarity = sorted(similarity, key=lambda x: -x[1])[:10];
    # filtSubs = [subsDict[lexelt.split('.')[0]][i[0]] for i in similarity]
    filtSubs = [subsDict[i[0]] for i in similarity]

    similarity = [i[1] for i in similarity]
    while len(similarity) < 10:
        similarity.append(0)
    numdump.append(similarity)

    print(lexelt, numsentences, filtSubs)
    # print('{} {} :: {}'.format(lexelt, numsentences, str(filtSubs).strip('[\']').replace('\', \'', ';')))
    bestFile.write('{} {} :: {}'.format(lexelt, numsentences, str(filtSubs[:3]).strip('[\']').replace('\', \'', ';')))
    bestFile.write('\n')
    ootFile.write('{} {} ::: {}'.format(lexelt, numsentences, str(filtSubs).strip('[\']').replace('\', \'', ';')))
    ootFile.write('\n')

    # for i, sub in enumerate(wvocab):
    #     if sub in w2vec:
    #         sIdx = w2vec[sub]
    #
    #         cos = cosine(wvecs[wIdx], wvecs[sIdx])
    #         # print('cosine({}, {}) = {}'.format(target, sub, cos))
    #         similarity.append((i, cos))
    #     # else:
    # # print('substitute word not in vocab: {}'.format(sub))
    #
    # # print(similarity)
    # filtSubs = [wvocab[i[0]] for i in sorted(similarity, key=lambda x: -x[1])[:11]]
    #
    # print(filtSubs)



    # if numsentences == 10:
    #     break

numdump2 = np.matrix(numdump)
np.save('../data/2/np.dump', numdump2)
