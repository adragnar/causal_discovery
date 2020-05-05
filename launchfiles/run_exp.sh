#!/bin/bash

#Make expieriment directories/files
expdir="/scratch/hdd001/home/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"

#Set cluster parameters
max_proc=60

#Set Experiment Type
dtype="adult"  #adult, german
exptype="single"  #all_combos, single

early_stopping=1
reduce_dsize=2000
binarize=1  #0, 1
takeout_envs=1
eq_estrat=-1  #-1, #samples_wanted
seeds=(1000 8079 52 147 256 784 990 587 304 737)


#Experiment Hyperparameters
if [ $dtype == "adult" ]
then
    data="~/causal_discovery/data/adult.csv"
    alphas="100"  #'list-of-vals' or 'range-start-stop-step'
    ft_combos=('12')

    #Only some environments binarized
    if [ $binarize == 0 ]
    then
        env_vars=("workclass" "native-country" "occupation" "marital-status" "relationship")
    fi
    if [ $binarize == 1 ]
    then
        env_vars=("workclass" "native-country" "occupation" "marital-status" "relationship")
    fi
fi

if [ $dtype == "german" ]
then
    data="~/causal_discovery/data/germanCredit.csv"
    alphas='100'  #"range-0.5-4.0-0.01"
    ft_combos=('' '1' '2' '12')

    #Only some environments binarized
    if [ $binarize == 0 ]
    then
        env_vars=('Purpose' 'Housing' 'Telephone' 'Property')  # 'OtherInstallmentPlans'  'Foreign' 'Savings' 'Personal' 'OtherDebtors')
    fi
    if [ $binarize == 1 ]
    then
        env_vars=('Purpose' 'Telephone' 'Housing' 'Property')
    fi
fi

#Generate the commandfile
for s in ${seeds[*]}
do
  for f_eng in ${ft_combos[*]}
  do
      python setup_params.py $alphas $f_eng $data $expdir $cmdfile ${env_vars[@]} -envcombos $exptype -early_stopping $early_stopping -reduce_dsize $reduce_dsize -binarize $binarize -takeout_envs $takeout_envs -eq_estrat $eq_estrat -seed $s
  done
done

#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

cmd=( $cmd )
num_tokens=${#cmd[@]}
xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile




#Note - must be run from the causal_discovery directory
