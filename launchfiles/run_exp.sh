#!/bin/bash

#Make expieriment directories/files
expdir="/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"

#Set cluster parameters
max_proc=32

#Set Experiment Type
dtype="adult"  #adult, german
exptype="single"  #all_combos, single
early_stopping=0
reduce_dsize=-1

#Experiment Hyperparameters
if [ $dtype == "adult" ]
then
    data="~/causal_discovery/data/adult.csv"
    alphas=(0.01 0.000001)
    ft_combos=('12')
    env_vars=("workclass" "native-country" "occupation" "marital-status" "education" "relationship")
fi

if [ $dtype == "german" ]
then
    data="~/causal_discovery/data/germanCredit.csv"
    alphas=(0.1 0.01 0.000001)
    ft_combos=('1' '2' '12')
    env_vars=('Purpose' 'Savings' 'Personal' 'OtherDebtors' 'Property' 'OtherInstallmentPlans' 'Housing' 'Foreign')
fi

#Generate the commandfile
for a in ${alphas[*]}
do
    for f_eng in ${ft_combos[*]}
    do
        python setup_params.py $a $f_eng $data $expdir $cmdfile ${env_vars[@]} -envcombos $exptype -early_stopping $early_stopping -reduce_dsize $reduce_dsize
    done
done

#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

cmd=( $cmd )
num_tokens=${#cmd[@]}
xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile




#Note - must be run from the causal_discovery directory
