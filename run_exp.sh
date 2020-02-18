#!/bin/bash

#Make expieriment directories
expdir="/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
logdir="$expdir/logs"
mkdir -p $logdir
data="~/causal_discovery/data/adult.csv"

#Set cluster parameters
cmdfile="$expdir/cmdfile.sh"
max_proc=32

#Experiment parameters
alphas=(0.00001)
ft_combos=('12')

env_vars=("workclass" "native-country" "occupation" "marital-status" "education")
#echo "${env_vars[@]}"

#Generate experiment comamnds
exptype="all_combos"

if [ $exptype == "single_combo" ]
then
    for a in ${alphas[*]}
    do
        for f_eng in ${ft_combos[*]}
        do
            subsetsfname="$expdir/${a}_${f_eng}_acc_subsets.txt"
            featuresfname="$expdir/${a}_${f_eng}_acc_features.txt"
            loggerfname="$logdir/${a}_${f_eng}_logging.csv"
            cmd="main.py $a $f_eng $data $subsetsfname $featuresfname $loggerfname ${env_vars[@]}"
            echo $cmd
        done
    done > $cmdfile
fi

if [ $exptype == "all_combos" ]
then
    for a in ${alphas[*]}
    do
        for f_eng in ${ft_combos[*]}
        do
            subsetsfname="$expdir/${a}_${f_eng}_acc_subsets.txt"
            featuresfname="$expdir/${a}_${f_eng}_acc_features.txt"
            loggerfname="$logdir/${a}_${f_eng}_logging.csv"
            srun --mem=16G -p cpu python setup_params.py $a $f_eng $data $subsetsfname $featuresfname $cmdfile $loggerfname ${env_vars[@]}
        done
    done
fi

#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

cmd=( $cmd )
num_tokens=${#cmd[@]}
xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile