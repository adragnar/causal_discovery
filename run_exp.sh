#!/bin/bash

#Make expieriment directories
expdir="/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
data="~/causal_discovery/data/adult.csv"

#Set cluster parameters
cmdfile="$expdir/cmdfile.sh"
max_proc=32

#Experiment parameters
alphas=(0.01)
ft_combos=('1' '2' '12')
env_vars=("occupation" "workclass" "native-country" "education" "marital-status")
#echo "${env_vars[@]}"

#Generate experiment comamnds
for a in ${alphas[*]}
do
    for f_eng in ${ft_combos[*]}
    do
        subsetsfname="$expdir/${a}_${f_eng}_acc_subsets.txt"
        featuresfname="$expdir/${a}_${f_eng}_acc_features.txt"
        cmd="python main.py $a $f_eng $data $subsetsfname $featuresfname ${env_vars[@]}"
        echo $cmd
    done
done > $cmdfile


#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

cmd=( $cmd )
num_tokens=${#cmd[@]}
xargs -n $num_tokens -P $max_proc srun --mem=16G -p cpu < $cmdfile