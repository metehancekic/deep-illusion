#!/bin/bash 

# # ./shell2.sh | tee shell2_out.txt
# pid=436088

# # check if the process is finished or not
# while [ -d /proc/$pid ] ; do
#     sleep 1
# done

#export CUDA_VISIBLE_DEVICES="1"

dataset=mnist
loss_function=cross_entropy

COMMAND="python test_AT.py  \
--dataset=$dataset  \
--tr_attack=FGSM  \
--loss_function=$loss_function" 
echo $COMMAND
eval $COMMAND


COMMAND="python test_AT.py  \
--dataset=$dataset  \
--tr_attack=RFGSM  \
--loss_function=$loss_function"

echo $COMMAND
eval $COMMAND

COMMAND="python test_AT.py  \
--dataset=$dataset  \
--tr_attack=PGD  \
--loss_function=$loss_function"  
echo $COMMAND
eval $COMMAND

