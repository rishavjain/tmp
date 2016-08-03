#!/bin/bash

EVAL=/home/cop15rj/rishav-msc-project/evaluate/lexsub-master
DATA=/data/cop15rj/lexsub/1
OUT=/data/cop15rj/lexsub/1.2
LOG=/data/cop15rj/lexsub/1.2
SCRIPTS=/home/cop15rj/rishav-msc-project/code
export PYTHONPATH='/home/cop15rj/rishav-msc-project/*:/home/cop15rj/rishav-msc-project/evaluate/lexsub-master/*:'

mkdir -p $OUT

#CMD1="python $EVAL/jcs/text2numpy.py $DATA/lexsub_word_embeddings $OUT/vecs"
##echo qsub -l mem=48G -l rmem=16G -j y -o $LOG/vecs.log -N text2numpy-vecs $SCRIPTS/redirect.bash $CMD1
#echo $CMD1
#$CMD1
#
#CMD2="python $EVAL/jcs/text2numpy.py $DATA/lexsub_context_embeddings $OUT/contexts"
##echo qsub -l mem=48G -l rmem=16G -j y -o $LOG/contexts.log -N text2numpy-contexts $SCRIPTS/redirect.bash $CMD2
#echo $CMD2
#$CMD2

echo python $EVAL/jcs/jcs_main.py --inferrer emb -vocabfile $EVAL/datasets/ukwac.vocab.lower.min100 -testfile $EVAL/datasets/lst_all.preprocessed -testfileconll $EVAL/datasets/lst_all.conll -candidatesfile $EVAL/datasets/lst.gold.candidates -embeddingpath $OUT/vecs -embeddingpathc $OUT/contexts -contextmath mult --debug -resultsfile $OUT/results
python $EVAL/jcs/jcs_main.py --inferrer emb -vocabfile $EVAL/datasets/ukwac.vocab.lower.min100 -testfile $EVAL/datasets/lst_all.preprocessed -testfileconll $EVAL/datasets/lst_all.conll -candidatesfile $EVAL/datasets/lst.gold.candidates -embeddingpath $OUT/vecs -embeddingpathc $OUT/contexts -contextmath mult --debug -resultsfile $OUT/results

#echo python $EVAL/jcs/evaluation/lst/lst_gap.py $EVAL/datasets/lst_all.gold $OUT/results.ranked $OUT/gap no-mwe
#python $EVAL/jcs/evaluation/lst/lst_gap.py $EVAL/datasets/lst_all.gold $OUT/results.ranked $OUT/gap no-mwe
