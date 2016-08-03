from utils import readconf
import sys
import numpy as np
from pprint import pformat
import pickle
from scipy.spatial.distance import cosine

params = readconf(sys.argv[1])


def readembeddings():
    wvecs = np.load(params['wvecs'])
    wvocab = open(params['wvocab']).read().split()
    w2vec = {w: i for i, w in enumerate(wvocab)}

    cvecs = np.load(params['cvecs'])
    cvocab = open(params['cvocab']).read().split()
    c2vec = {w: i for i, w in enumerate(cvocab)}

    return wvocab, wvecs, w2vec, cvocab, cvecs, c2vec


def targetcontext(conll, tIdx):
    root = [0, '*root*', None, None, None, -1, 'rroot']
    tokens = [root] + conll

    tIdx += 1

    contexts = []
    for tok in tokens:
        par_ind = int(tok[5])
        par = tokens[par_ind]
        m = tok[1]
        mIdx = int(tok[0])

        rel = tok[6]
        if rel == 'prep': continue  # this is the prep. we'll get there (or the PP is crappy)

        if rel == 'pobj' and par[0] != 0:

            ppar = tokens[int(par[5])]
            rel = "%s:%s" % (par[6], par[1])
            h = ppar[1]
            hIdx = int(ppar[0])
        else:
            h = par[1]
            hIdx = int(par[0])

        if h != '*root*' and hIdx == tIdx:
            contexts += ["_".join((rel, m))]
        if mIdx == tIdx:
            contexts += ["I_".join((rel, h))]

    return contexts


def readtaskdata():
    taskdata = pickle.load(open(params['testdata'], 'rb'))

    print(type(taskdata), len(taskdata))
    return taskdata

def add(tIdx, sIdx, contexts):
    cos = cosine(wvecs[tIdx], wvecs[sIdx])

    ncontexts = 0
    for context in contexts:
        if context in c2vec:
            cIdx = c2vec[context]
            cos += cosine(cvecs[cIdx], wvecs[sIdx])
            ncontexts += 1

    cos = (cos/(ncontexts + 1.0))

    return cos

def predict(taskdata, wvocab, wvecs, w2vec, cvocab, cvecs, c2vec):
    GAP = []
    for idx in taskdata:
    # for idx in ['670']:
        item = taskdata[idx]
        target = item['t']

        if target not in w2vec:
            print('target not in vocab:', target)
            continue

        tIdx = w2vec[target]

        contexts = targetcontext(item['conll'], item['tIdx'])

        gap = 0
        lexsub = {}

        dist = []
        for sub in item['subs']:
            if sub[0] not in w2vec:
                continue

            sIdx = w2vec[sub[0]]


            dist.append([sub, add(tIdx, sIdx, contexts)])

        dist = sorted(dist, key=lambda x: -x[1])

        x = []
        gold = [i[0] for i in item['gold']]
        gweights = [int(i[1]) for i in item['gold']]

        for i in dist:
            if i[0] in gold:
                x.append(int(gweights[gold.index(i[0])]))
            else:
                x.append(0)

        gap = 0
        for i in range(0, len(dist)):
            if dist[i][0] in gold:
                p = 0
                for k in range(0, i+1):
                    p += x[k]
                p /= (i+1)
                gap += p

        R = 0
        for i in range(0, len(gold)):
            p = 0
            for k in range(0, i+1):
                p += gweights[k]
            p /= (i+1)
            R += p

        GAP.append(gap/R)
        lexsub[idx] = dist

        # print(i, gap)
        #
        # print(item['cstr'])
        # print(item['gold'])
        # print(dist)
    return GAP, lexsub

print('params', ':', pformat(params))
wvocab, wvecs, w2vec, cvocab, cvecs, c2vec = readembeddings()
taskdata = readtaskdata()
GAP, lexsub = predict(taskdata, wvocab, wvecs, w2vec, cvocab, cvecs, c2vec)

pickle.dump(lexsub, open(params['out'], 'wb'))

GAP = np.array(GAP).mean()
print('GAP =', GAP)


"""
import pickle
import xml.etree.ElementTree as ET
import re
import os
import numpy as np
from scipy.spatial.distance import cosine
from extract_deps import read_conll
import heapq



TARGETS_FILE = '../evaluate/data/targets.txt'
TARGETS_FILE = '/home/cop15rj/rishav-msc-project/evaluate/data/targets.txt'
# SUBSTITUTES_FILE = '../evaluate/data/substitutes.data'
SUBSTITUTES_FILE = '../data/2/wv'
SUBSTITUTES_FILE = '/data/cop15rj/lexsub/1/wv'
# SUBSTITUTES_FILE = '/home/cop15rj/rishav-msc-project/evaluate/data/substitutes.data'


INPUT_CONLL_FILE = '../evaluate/data/sentences.txt.conll'
INPUT_CONLL_FILE = '/home/cop15rj/rishav-msc-project/evaluate/data/sentences.txt.conll'
# INPUT_FILE = 'evaluate/taskdata/trial/lexsub_trial.xml'

OUTPUT_FILE_BEST = '../data/2/best.txt'
OUTPUT_FILE_BEST = '/data/cop15rj/lexsub/1/best.txt'

OUTPUT_FILE_OOT = '../data/2/oot.txt'
OUTPUT_FILE_OOT = '/data/cop15rj/lexsub/1/oot.txt'

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
        print('target word not in vocab: {}'.format(target))

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

            # heapq.heappush(similarity, (-cos/(numcontexts + 1.0), subsDict[i]))
            similarity.append((cos/(numcontexts + 1.0), subsDict[i]))
        # else:
            # print('substitute word not in vocab: {}'.format(sub))

    similarity = heapq.nlargest(10, similarity, key=lambda x: x[0])

    # print(similarity)
    # similarity = sorted(similarity, key=lambda x: -x[1])[:10];
    # filtSubs = [subsDict[lexelt.split('.')[0]][i[0]] for i in similarity]
    filtSubs = [i[1] for i in similarity]
    # filtSubs = [heapq.heappop(similarity)[1] for i in range(10)]

    # similarity = [i[1] for i in similarity]
    # while len(similarity) < 10:
    #     similarity.append(0)
    # numdump.append(similarity)

    print(lexelt, numsentences, contexts, filtSubs)
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

# numdump2 = np.matrix(numdump)
# np.save('../data/2/np.dump', numdump2)
"""