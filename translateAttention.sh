#!/bin/sh

#$ -M cristinae@dfki.de
#$ -m eas
#$ -V
#$ -S /bin/bash

#$ -cwd
#$ -wd /home/cristinae/sge/

#$ -o /home/cristinae/sge/soft/devnematus/tl.o.txt
#$ -e /home/cristinae/sge/soft/devnematus/tl.e.txt

#$ -l gpu=1
# -l h=exs-91208.sb.dfki.de

export CUDA_HOME=/usr/local/cuda-7.5/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/raid/bin/cudnn-v2/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH
export PATH=$PATH:${CUDA_HOME}/bin


# inputs
#model=$1  #ULL al exe tb
src=l
tgt=L2
model='model_L1L2bpetrueLs_v60k.iter650000.npz'
#testPath='/home/cristinae/sge/semeval/task1MT/'


 theano device
device=gpu

# path to this nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/cristinae/sge/soft/devnematus
testPath=$nematus

#data=/raid/data/europarl/tests
#test=$data/newstest2013.tc.bpe.e
#test=/home/cristinae/sge/soft/devnematus/trial.l.L1
#testContext=/home/cristinae/sge/soft/devnematus/trial.sum.l.L1.att
#trad=newstest2013.$model.tc.${src}2${tgt}
#trad=/home/cristinae/sge/soft/devnematus/trial.sum.l.L2

test=$testPath/$1
trad=$test.L2
testContext=$test.'ctx'

THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m /home/cristinae/sge/semeval/task1/MTmodels/$model \
     -i $test \
     -o $trad \
     --output_context $testContext \
     -n -p 1
