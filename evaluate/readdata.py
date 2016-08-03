from common.utils import create_dirs
import os
import pickle
import pprint
import xml.etree.ElementTree
import sys
import re

INPUT_FILES = ['taskdata/trial/lexsub_trial.xml', 'taskdata/test/lexsub_test.xml']
TARGETS_OUTPUT = 'data/targets.txt'
SENTENCES_OUTPUT = 'data/sentences.txt'


INPUT_FILES = [os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), INPUT_FILE) for INPUT_FILE in INPUT_FILES]
print('INPUT_FILES', ':', INPUT_FILES)

TARGETS_OUTPUT = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), TARGETS_OUTPUT)
create_dirs(TARGETS_OUTPUT)

SENTENCES_OUTPUT = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), SENTENCES_OUTPUT)
create_dirs(SENTENCES_OUTPUT)

if __name__ == '__main__':

    targetsFile = open(TARGETS_OUTPUT, 'w')
    sentencesFile = open(SENTENCES_OUTPUT, 'w')

    for INPUT_FILE in INPUT_FILES:
        xmlTree = xml.etree.ElementTree.parse(INPUT_FILE)
        root = xmlTree.getroot()

        print('root.tag', ':', root.tag)

        for lexelt in root:
            print('lexelt.item', ':', lexelt.attrib['item'])

            for instance in lexelt:
                print('instance.id', ':', instance.attrib['id'])

                for context in instance:
                    # print('context', ':', xml.etree.ElementTree.tostring(context))
                    contextStr = xml.etree.ElementTree.tostring(context).decode()
                    contextStr = contextStr.strip().replace('<context>', '').replace('</context>', '')

                    print('contextStr', ':', contextStr)

                    t = re.findall('<head>.+</head>', contextStr)[0]

                    idx = (contextStr.split(' ')).index(t) + 1

                    print('t', ':', t.replace('<head>', '').replace('</head>', ''))

                    contextStr = contextStr.replace(t, t.replace('<head>', '').replace('</head>', ''))
                    # print('contextStr', ':', contextStr)

                    print(lexelt.attrib['item'], t.replace('<head>', '').replace('</head>', ''), idx, file=targetsFile)
                    print(contextStr, file=sentencesFile)

        #         break
        #     break
        # break

    targetsFile.close()
    sentencesFile.close()
