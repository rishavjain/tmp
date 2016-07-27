"""

cat data\ukwac_10M.conll evaluate\data\sentences.txt.conll > data\combined10.conll

cut -f 2 data\combined10.conll | python word2vecf_scripts\vocab.py 100 > tmp\combined10_vocab.txt

cat data\combined10.conll | python word2vecf_scripts\extract_deps.py conll tmp\combined10_vocab.txt 100 > tmp\combined10_dep.txt

word2vecf\count_and_filter -train tmp\combined10_dep.txt -cvocab tmp\combined_context.txt -wvocab tmp\combined_vocab.txt -min-count 100

word2vecf\word2vecf -train tmp\combined10_dep.txt -cvocab tmp\combined_context.txt -wvocab tmp\combined_vocab.txt -output tmp\dim600vecs -dumpcv tmp\dim600contexts -size 600 -negative 15 -threads 32

python word2vecf_scripts\vecs2nps.py tmp\dim600vecs data\vecs
python word2vecf_scripts\vecs2nps.py tmp\dim600contexts data\contexts

"""