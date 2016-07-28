import sys
from collections import Counter

wc = Counter()
thr = int(sys.argv[1])
l = []

try:
    sys.stdin = open(sys.argv[2], 'r')
except IndexError:
    print('using console input', file=sys.stderr)

for i, w in enumerate(sys.stdin):
    # print('D:', i, w)
    if i % 1000000 == 0:
        # if i > 10000000: break
        print(i, len(wc), file=sys.stderr)
        wc.update(l)
        l = []
    l.append(w.strip().lower())
wc.update(l)

# tag = '<~unknown~>'
# tagcount = 0
for w, c in [(w, c) for w, c in wc.items()]:
    if c < thr or w == '':
        # tagcount += c
        wc.pop(w)

# wc[tag] = tagcount

# sorting the counter in descending order of count
for w, c in sorted([(w, c) for w, c in wc.items()], key=lambda x: -x[1]):
    print("\t".join([w, str(c)]))
