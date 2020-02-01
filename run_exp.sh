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
env_vars=("occupation" "workclass" "native-country" "education" "marital-status")
#echo "${env_vars[@]}"

#Generate experiment comamnds
for a in ${alphas[*]}
do
    subsetsfname="$expdir/${a}_acc_subsets.txt"
    featuresfname="$expdir/${a}_acc_features.txt"
    srun --mem=16G -p cpu python setup_params.py $a $data $subsetsfname $featuresfname $cmdfile "${env_vars[@]}"
done


#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

xargs -P $max_proc srun --mem=16G -p cpu < $cmdfile
echo cmds_sent