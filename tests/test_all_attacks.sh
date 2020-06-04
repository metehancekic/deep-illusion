#!/bin/bash 

# ./shell2.sh | tee shell2_out.txt
pid=436088

# check if the process is finished or not
while [ -d /proc/$pid ] ; do
    sleep 1
done

#export CUDA_VISIBLE_DEVICES="1"

dataset=mnist

COMMAND="python test_attacks.py  \
--dataset=$dataset  \
--attack_method=FGSM"  
echo $COMMAND
eval $COMMAND

COMMAND="python test_attacks.py  \
--dataset=$dataset  \
--attack_method=RFGSM"  
echo $COMMAND
eval $COMMAND

COMMAND="python test_attacks.py  \
--dataset=$dataset  \
--attack_method=BIM"  
echo $COMMAND
eval $COMMAND

COMMAND="python test_attacks.py  \
--dataset=$dataset  \
--attack_method=BIM_EOT"  
echo $COMMAND
eval $COMMAND

COMMAND="python test_attacks.py  \
--dataset=$dataset  \
--attack_method=PGD"  
echo $COMMAND
eval $COMMAND

COMMAND="python test_attacks.py  \
--dataset=$dataset  \
--attack_method=PGD_EOT"  
echo $COMMAND
eval $COMMAND

COMMAND="python test_attacks.py  \
--dataset=$dataset  \
--attack_method=PGD_EOT_normalized"  
echo $COMMAND
eval $COMMAND

COMMAND="python test_attacks.py  \
--dataset=$dataset  \
--attack_method=PGD_EOT_sign"  
echo $COMMAND
eval $COMMAND




