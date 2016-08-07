#!/bin/bash

EVAL=/home/cop15rj/rishav-msc-project/evaluate/lexsub-master
DATA=/data/cop15rj/lexsub/2.1
DATA=/data/cop15rj/lexsub/3
OUT=/data/cop15rj/lexsub/2.1
OUT=/data/cop15rj/lexsub/3.1
LOG=/data/cop15rj/lexsub/2.1
LOG=/data/cop15rj/lexsub/3.1
SCRIPTS=/home/cop15rj/rishav-msc-project/code
#export PYTHONPATH='/home/cop15rj/rishav-msc-project/evaluate/lexsub-master/*:'

mkdir -p $OUT

#CMD1="python $EVAL/jcs/text2numpy.py $DATA/dim600vecs $OUT/vecs"
##echo qsub -l mem=48G -l rmem=16G -j y -o $LOG/vecs.log -N text2numpy-vecs $SCRIPTS/redirect.bash $CMD1
#echo $CMD1
##$CMD1
#
#CMD2="python $EVAL/jcs/text2numpy.py $DATA/dim600contexts $OUT/contexts"
##echo qsub -l mem=48G -l rmem=16G -j y -o $LOG/contexts.log -N text2numpy-contexts $SCRIPTS/redirect.bash $CMD2
#echo $CMD2
##$CMD2

CURR=$(pwd)

cd $EVAL

#echo python jcs/jcs_main.py --inferrer emb -vocabfile $EVAL/datasets/ukwac.vocab.lower.min100 -testfile $EVAL/datasets/lst_all.preprocessed -testfileconll $EVAL/datasets/lst_all.conll -candidatesfile $EVAL/datasets/lst.gold.candidates -embeddingpath $DATA/vecs -embeddingpathc $DATA/contexts -contextmath mult --debug -resultsfile $OUT/results
#python jcs/jcs_main.py --inferrer emb -vocabfile $EVAL/datasets/ukwac.vocab.lower.min100 -testfile $EVAL/datasets/lst_all.preprocessed -testfileconll $EVAL/datasets/lst_all.conll -candidatesfile $EVAL/datasets/lst.gold.candidates -embeddingpath $DATA/vecs -embeddingpathc $DATA/contexts -contextmath mult --debug -resultsfile $OUT/results

echo python jcs/evaluation/lst/lst_gap.py $EVAL/datasets/lst_all.gold $OUT/results.ranked $OUT/gap no-mwe
python jcs/evaluation/lst/lst_gap.py $EVAL/datasets/lst_all.gold $OUT/results.ranked $OUT/gap no-mwe

cd $CURR