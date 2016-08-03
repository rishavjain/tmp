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

"""
zcat ukwac100.conll.gz semeval-data/sentences.txt.conll.gz | gzip -c -v > data100.conll.gz

zcat /data/cop15rj/data100.conll.gz | cut -f 2 | python code/vocab.py 100 > /data/cop15rj/lexsub/1/vocab.txt

zcat /data/cop15rj/data100.conll.gz | python /home/cop15rj/rishav-msc-project/code/extract_deps.py conll /data/cop15rj/lexsub/1/vocab.txt 100 > /data/cop15rj/lexsub/1/data100.dep

/home/cop15rj/rishav-msc-project/word2vecf/count_and_filter -train /data/cop15rj/lexsub/1/data100.dep -cvocab /data/cop15rj/lexsub/1/cv -wvocab /data/cop15rj/lexsub/1/wv -min-count 100

/home/cop15rj/rishav-msc-project/word2vecf/word2vecf -train /data/cop15rj/lexsub/1/data100.dep -cvocab /data/cop15rj/lexsub/1/cv -wvocab /data/cop15rj/lexsub/1/wv -output /data/cop15rj/lexsub/1/dim600vecs -dumpcv /data/cop15rj/lexsub/1/dim600contexts -size 600 -negative 15 -threads 32

python /home/cop15rj/rishav-msc-project/code/vecs2nps.py /data/cop15rj/lexsub/1/dim600vecs /data/cop15rj/lexsub/1/vecs
python /home/cop15rj/rishav-msc-project/code/vecs2nps.py /data/cop15rj/lexsub/1/dim600contexts /data/cop15rj/lexsub/1/contexts

qsub -l mem=48G -l rmem=16G -m bea -M rjain2@sheffield.ac.uk -j y -o /data/cop15rj/log-qsub -N automate /data/cop15rj/run-code.bash
"""