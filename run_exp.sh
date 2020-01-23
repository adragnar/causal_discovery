#!/bin/bash

#Make expieriment directories
expdir="/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
data="~/causal_discovery/data/adult.csv"

#Set cluster parameters
cmdfile="$expdir/cmdfile.sh"
max_proc=20

#Experiment parameters
alphas=(0.01 0.05 0.10)

#Generate experiment comamnds
for a in ${alphas[*]}
do
    subsetsfname="$expdir/${a}_acc_subsets.txt"
    featuresfname="$expdir/${a}_acc_features.txt"
    cmd="python main.py $a $data $subsetsfname $featuresfname"
    echo $cmd
done > $cmdfile


#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

cmd=( $cmd )
num_tokens=${#cmd[@]}
echo $cmd
echo $num_tokens
xargs -n $num_tokens -P $max_proc srun --mem=16G -p cpu < $cmdfile
echo cmds_sent