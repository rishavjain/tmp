"""
set path=%path%;\\studata05\home\CO\Cop15rj\cygwin\bin

cat .\data\1\ukwac_10M.conll .\evaluate\data\sentences.txt.conll > .\data\1\combined10.conll

cut -f 2 .\data\2\combined10.conll | python code\vocab.py 100 > .\data\2\vocab.txt

cat .\data\2\combined10.conll | python code\extract_deps.py conll .\data\2\vocab.txt 100 > .\data\2\combined10.dep

word2vecf\count_and_filter -train .\data\2\combined10.dep -cvocab .\data\2\cv -wvocab .\data\2\wv -min-count 100

word2vecf\word2vecf -train .\data\2\combined10.dep -cvocab .\data\2\cv -wvocab .\data\2\wv -output .\data\2\dim600vecs -dumpcv .\data\2\dim600contexts -size 600 -negative 15 -threads 32

python code\vecs2nps.py .\data\2\dim600vecs .\data\2\vecs
python code\vecs2nps.py .\data\2\dim600contexts .\data\2\contexts

"""