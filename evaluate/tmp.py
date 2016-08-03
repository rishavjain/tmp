from pprint import pprint
import xml.etree.ElementTree as xml
import re
import random
import pickle

INPUT_FILES = ['./taskdata/trial/lexsub_trial.xml',
               './taskdata/test/lexsub_test.xml']

CONLL_FILE = 'sentences.txt.conll'

GOLD_FILE = 'gold.txt'

OUTPUT_FILE = './semeval.data'

def read_conll():
    conll = []
    item = []
    for line in open(CONLL_FILE):
        line = line.strip()

        if not line:
            if len(item) > 1:
                conll += [item,]
                item = []
            continue

        item += [line.split('\t'),]

    return conll

def read_gold():
    substitutes = {}
    gold = {}

    for line in open(GOLD_FILE):
        line = line.strip()

        if not line:
            continue

        idx = line.split()[1]
        word = line.split()[0]

        subs = [tuple(x.split()) for x in line.split('::')[1].strip().split(';') if len(x) > 0]
        subs = [x for x in subs if len(x)==2]

        if not subs:
            continue

        gold[idx] = subs

        if word not in substitutes:
            substitutes[word] = set([x[0] for x in gold[idx]])
        else:
            substitutes[word] |= set([x[0] for x in gold[idx]])

    return gold, substitutes

def match_conll(s, conll):
    for line in conll:
        w = line[1]
        if w.isalnum() and w not in s:
            return False
    return True

def read_input_xml():
    data = {}
    conllId = 0

    for INPUT_FILE in INPUT_FILES:
        xmlTree = xml.parse(INPUT_FILE)

        root = xmlTree.getroot()

        for item in root:
            for instance in item:
                for context in instance:
                    idx = instance.attrib['id']

                    if idx not in gold:
                        continue

                    data[idx] = {}

                    contextStr = xml.tostring(context).decode().strip().replace('<context>', '').replace('</context>', '')
                    data[idx]['cstr'] = contextStr
                    data[idx]['lexelt'] = item.attrib['item']

                    target = re.findall('<head>.+</head>', contextStr)[0].replace('<head>', '').replace('</head>', '')
                    data[idx]['t'] = target.lower()

                    data[idx]['gold'] = gold[idx]
                    data[idx]['subs'] = substitutes[data[idx]['lexelt']]

                    data[idx]['conll'] = []
                    if match_conll(contextStr, conlls[conllId]):
                        data[idx]['conll'] += [conlls[conllId],]
                        conllId = (conllId + 1)%len(conlls)

                        while match_conll(contextStr, conlls[conllId]):
                            data[idx]['conll'] += [conlls[conllId],]
                            conllId = conllId + 1
                    else:
                        while not match_conll(contextStr, conlls[conllId]):
                            conllId = (conllId + 1)%len(conlls)

                        data[idx]['conll'] += [conlls[conllId],]
                        conllId = (conllId + 1)%len(conlls)

                        while match_conll(contextStr, conlls[conllId]):
                            data[idx]['conll'] += [conlls[conllId],]
                            conllId = conllId + 1

                    W = contextStr.split()
                    tIdx = [i for i, x in enumerate(W) if '<head>' in x][0]

                    data[idx]['tIdx'] = None

                    for conll in data[idx]['conll']:
                        w = [l[1] for l in conll]

                        wIdx = None
                        for op in [-2,-1,1,2]:
                            if tIdx+op > 0 and tIdx+op < len(W) and W[tIdx+op] in w:
                                wIdx = w.index(W[tIdx+op])
                                if wIdx-op < len(w) and wIdx-op > 0 and w[wIdx-op] == target:
                                    wIdx = wIdx-op
                                    break

                        if wIdx is not None:
                            data[idx]['tIdx'] = wIdx
                            data[idx]['conll'] = conll
                            break

                    if data[idx]['tIdx'] is None:
                        data.pop(idx)
                    
    return data

conlls = read_conll()
print('len(conlls)',':',len(conlls))

gold, substitutes = read_gold()
print('len(gold)',':',len(gold))
print('len(substitutes)',':',len(substitutes))

data = read_input_xml()
print('len(data)',':',len(data))

pickle.dump(data, open(OUTPUT_FILE, 'wb'))