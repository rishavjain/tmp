#!/bin/bash

HOME=/home/cop15rj/rishav-msc-project
CODE=/home/cop15rj/rishav-msc-project/code
OUT=/data/cop15rj/lexsub/2

# User specific aliases and functions
module load apps/python/anaconda3-2.5.0
module load apps/java/1.8u71
export PYTHONPATH='/home/cop15rj/rishav-msc-project/*:'
export PYTHONIOENCODING=iso-8859-1
CPU=$(grep -c ^processor /proc/cpuinfo)


STARTTIME=$(date +%s)
echo HOME = $HOME
echo CODE = $CODE
echo OUT = $OUT
echo "CPU = $((CPU))"
mkdir -p $OUT
ENDTIME=$(date +%s)
echo "$(date): It takes $(($ENDTIME - $STARTTIME)) seconds to test the variables"

#STARTTIME=$(date +%s)
#zcat ukwac100.conll.gz semeval-data/sentences.txt.conll.gz | gzip -c -v > data100.conll.gz
#ENDTIME=$(date +%s)
#echo "$(date): It takes $(($ENDTIME - $STARTTIME)) seconds to complete 'concat' task..."

STARTTIME=$(date +%s)
zcat /data/cop15rj/data100.conll.gz | cut -f 2 | python $CODE/vocab.py 100 > $OUT/vocab.txt
ENDTIME=$(date +%s)
echo "$(date): It takes $(($ENDTIME - $STARTTIME)) seconds to complete 'vocab' task..."

STARTTIME=$(date +%s)
zcat /data/cop15rj/data100.conll.gz | python $CODE/extract_deps.py conll $OUT/vocab.txt 100 > $OUT/data100.dep
ENDTIME=$(date +%s)
echo "$(date): It takes $(($ENDTIME - $STARTTIME)) seconds to complete 'extract_deps' task..."

STARTTIME=$(date +%s)
$HOME/word2vecf/count_and_filter -train $OUT/data100.dep -cvocab $OUT/cv -wvocab $OUT/wv -min-count 100
ENDTIME=$(date +%s)
echo "$(date): It takes $(($ENDTIME - $STARTTIME)) seconds to complete 'count_and_filter' task..."

STARTTIME=$(date +%s)
$HOME/word2vecf/word2vecf -train $OUT/data100.dep -cvocab $OUT/cv -wvocab $OUT/wv -output $OUT/dim600vecs -dumpcv $OUT/dim600contexts -size 600 -negative 15 -threads $CPU
ENDTIME=$(date +%s)
echo "$(date): It takes $(($ENDTIME - $STARTTIME)) seconds to complete 'word2vecf' task..."

STARTTIME=$(date +%s)
python $CODE/vecs2nps.py $OUT/dim600vecs $OUT/vecs
ENDTIME=$(date +%s)
echo "$(date): It takes $(($ENDTIME - $STARTTIME)) seconds to complete 'vecs2nps' task..."

STARTTIME=$(date +%s)
python $CODE/vecs2nps.py $OUT/dim600contexts $OUT/contexts
ENDTIME=$(date +%s)
echo "$(date): It takes $(($ENDTIME - $STARTTIME)) seconds to complete 'vecs2nps' task..."
