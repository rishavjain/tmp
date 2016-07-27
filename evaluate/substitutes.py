"""
calculate possible substitutes from gold standard
"""

from common.utils import create_dirs
import os
import pickle
import pprint

GOLD_FILES = ('taskdata/trial/gold.trial', 'taskdata/scoring/gold')
TEXT_OUTPUT = 'data/substitutes.txt'
DICT_OUTPUT = 'data/substitutes.data'

subs = {}

if __name__ == '__main__':
    for GOLD_FILE in GOLD_FILES:
        goldFile = open(GOLD_FILE)

        for line in goldFile:
            line = line.strip()
            if len(line) > 0:
                # print(line)

                lineSplit = line.split('::')
                word = lineSplit[0].split('.')[0]
                _subs = {item.split()[0] for item in lineSplit[1].strip(' ;').split(';') if len(item.split()) == 2}

                if word in subs:
                    subs[word] |= _subs
                else:
                    subs[word] = _subs

                    # print(word, ':', subs[word])

    # print('substitutes', ':')
    # pprint.PrettyPrinter().pprint(subs)

    create_dirs(os.path.dirname(TEXT_OUTPUT))
    create_dirs(os.path.dirname(DICT_OUTPUT))

    outFile = open(TEXT_OUTPUT, 'w')
    for word in subs:
        outFile.write(' '.join([word, str(subs[word]).strip('{}\'').replace('\', \'', ' ')]))
        outFile.write('\n')
    outFile.close()

    pickle.dump(subs, open(DICT_OUTPUT, 'wb'))
