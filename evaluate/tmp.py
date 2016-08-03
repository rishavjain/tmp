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
    goldlines = {}

    for line in open(GOLD_FILE):
        line = line.strip()

        if not line:
            continue

        idx = line.split()[1]

        goldlines[idx] = [tuple(x.split()) for x in line.split('::')[1].strip().split(';') if len(x) > 0]

    return goldlines

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

                    if idx not in goldlines:
                        continue

                    data[idx] = {}

                    contextStr = xml.tostring(context).decode().strip().replace('<context>', '').replace('</context>', '')
                    data[idx]['cstr'] = contextStr
                    data[idx]['lexelt'] = item.attrib['item']

                    target = re.findall('<head>.+</head>', contextStr)[0].replace('<head>', '').replace('</head>', '')
                    data[idx]['t'] = target

                    data[idx]['subs'] = goldlines[idx]

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
                        if tIdx - 2 > 0 and W[tIdx-2] in w:
                            wIdx = w.index(W[tIdx-2])
                            if wIdx+2 < len(w) and w[wIdx+2] == target: wIdx = wIdx+2
                        elif tIdx - 1 > 0 and W[tIdx-1] in w:
                            wIdx = w.index(W[tIdx-1])
                            if wIdx+1 < len(w) and w[wIdx+1] == target: wIdx = wIdx+1
                        elif tIdx + 2 < len(W) and W[tIdx+2] in w:
                            wIdx = w.index(W[tIdx+2])
                            if wIdx-2 > 0 and w[wIdx-2] == target: wIdx = wIdx-2
                        elif tIdx + 1 < len(W) and W[tIdx+1] in w:
                            wIdx = w.index(W[tIdx+1])
                            if wIdx-1 > 0 and w[wIdx-1] == target: wIdx = wIdx-1

                        if wIdx is not None:
                            data[idx]['tIdx'] = wIdx
                            data[idx]['conll'] = conll
                            break

                    
    return data

conlls = read_conll()
print('len(conlls)',':',len(conlls))

goldlines = read_gold()
print('len(goldlines)',':',len(goldlines))

data = read_input_xml()
print('len(data)',':',len(data))

pickle.dump(data, open(OUTPUT_FILE, 'wb'))