#$ -M cristinae@dfki.de
#$ -m eas
#$ -V
#$ -S /bin/bash

# -cwd
#$ -wd /home/cristinae/sge/

#$ -o bpePrepSTSl2.o.txt
#$ -e bpePrepSTSl2.e.txt

#$ -l gpu=1

export CUDA_HOME=/usr/local/cuda-7.5/
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/raid/bin/cudnn-v2/cudnn-6.5-linux-x64-v2:$LD_LIBRARY_PATH
export PATH=$PATH:${CUDA_HOME}/bin


# Initialise
SRC="L1"
SRC="w"
TRG="L2"

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=62000


base=~/sge/soft/nematus
subword_nmt=$base/subword-nmt
data=~/sge/semeval/corpus/

corpus="/multiLingual50shuf.w"
dev='/multiLingual50shuf.dev.w'

test='/home/cristinae/sge/soft/devnematus/STS.input.track6.tr-en.trg.tr.txt.google.trad2enw'
test1='/home/cristinae/sge/semeval/task1MT/ar.train.1_fold/cris/ar.src.input.0.txt.w.trad2en'
test2='/home/cristinae/sge/semeval/task1MT/ar.train.1_fold/cris/ar.trg.input.0.txt.w.trad2en'
test3='/home/cristinae/sge/semeval/task1MT/en_ar.train.1_fold/cris/en_ar.src.input.0.txt.w.trad2en'
test4='/home/cristinae/sge/semeval/task1MT/en_ar.train.1_fold/cris/en_ar.trg.input.0.txt.w.trad2en'
test5='/home/cristinae/sge/semeval/task1MT/en_es.train.1_fold/cris/en_es.src.input.0.txt.w.trad2en'
test6='/home/cristinae/sge/semeval/task1MT/en_es.train.1_fold/cris/en_es.trg.input.0.txt.w.trad2en'
test7='/home/cristinae/sge/semeval/task1MT/en.test.1_fold/cris/en.src.input.0.txt.w.trad2es'
test8='/home/cristinae/sge/semeval/task1MT/en.test.1_fold/cris/en.trg.input.0.txt.w.trad2es'
test9='/home/cristinae/sge/semeval/task1MT/en.train.1_fold/cris/en.src.input.0.txt.w.trad2es'
test10='/home/cristinae/sge/semeval/task1MT/en.train.1_fold/cris/en.trg.input.0.txt.w.trad2es'
test12='/home/cristinae/sge/semeval/task1MT/es.train.1_fold/cris/es.src.input.0.txt.w.trad2en'
test13='/home/cristinae/sge/semeval/task1MT/es.train.1_fold/cris/es.trg.input.0.txt.w.trad2en'
test14='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track1.ar-ar.src.ar.txt.w.trad2en'
test15='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track1.ar-ar.trg.ar.txt.w.trad2en'
test16='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track2.ar-en.src.en.txt.w.trad2en'
test17='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track2.ar-en.trg.ar.txt.w.trad2en'
test18='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track3.es-es.src.es.txt.w.trad2en'
test19='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track3.es-es.trg.es.txt.w.trad2en'
test20='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track4a.es-en.src.en.txt.w.trad2en'
test21='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track4a.es-en.trg.es.txt.w.trad2en'
test22='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track4b.es-en.src.en.txt.w.trad2en'
test23='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track4b.es-en.trg.es.txt.w.trad2en'
test24='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track5.en-en.src.en.txt.w.trad2es'
test25='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track5.en-en.trg.en.txt.w.trad2es'
#test26='/home/cristinae/sge/semeval/task1MT/STS2017.eval/cris/STS.input.track6.tr-en.src.en.txt.w.trad2en'

#python $base/data/build_dictionary.py $data/$corpus.$SRC $data/$corpus.$TRG

# train BPE
#cat $data/$dev.$SRC $data/$dev.$TRG | python $subword_nmt/learn_bpe.py -s $bpe_operations > modelN2/$SRC$TRG.bpe
#cat $data/$corpus.$SRC $data/$corpus.$TRG | python $subword_nmt/learn_bpe.py -s $bpe_operations > modelNou/$SRC$TRG.bpe

# apply BPE
#for prefix in $test9 $test10 $test2 $test3 $test4 $test5 $test6 $test7 $test8 $test1\
#		     $test11 $test12 $test13 $test14 $test15 $test16 $test17 $test18 $test19 $test20\
#		     $test21 $test22 $test23 $test24 $test25 
for prefix in $test  	      #for prefix in  $dev	      
do
#    $subword_nmt/apply_bpe.py -c $base/modelNou/$SRC$TRG.bpe < $prefix.$SRC > $prefix.bpe.$SRC
        $subword_nmt/apply_bpe.py -c $base/modelNou/L1L2.bpe < $prefix.$SRC > $prefix.bpe.$SRC 
#    $subword_nmt/apply_bpe.py -c modelNou/$SRC$TRG.bpe < $data/$prefix.$TRG > $data/$prefix.bpe.$TRG
done

# build network dictionary
# python $base/data/build_dictionary.py $data/$corpus.bpe.$SRC $data/$corpus.bpe.$TRG &

