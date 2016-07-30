
import sys
import pprint

f1 = open('./data/1/combined10.conll')

f2 = sys.stdin

s = []

for f in [f1, f2]:
    for line in f:
        if '7138' in line:
            s.append(line.strip())

print(pprint.pprint(s))
