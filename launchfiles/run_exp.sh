#!/bin/bash

#Make expieriment directories/files
expdir="/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"
data="~/causal_discovery/data/adult.csv"

#Set cluster parameters
max_proc=32

#Set Experiment Type
exptype="all_combos"
testing=1

#Experiment Hyperparameters
alphas=(0.01 0.00001)
ft_combos=('1' '2' '12')
env_vars=("workclass" "native-country" "occupation" "marital-status" "education")

#Generate the commandfile
for a in ${alphas[*]}
do
    for f_eng in ${ft_combos[*]}
    do
        python "$(dirname "$(pwd)")/setup_params.py" $a $f_eng $data $expdir $cmdfile ${env_vars[@]} --envcombos 0 --testing $testing
    done
done






#Note - must be run from the causal_discovery directory