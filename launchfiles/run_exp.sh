#!/bin/bash

#Make expieriment directories/files
expdir="/scratch/hdd001/home/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"

#Set cluster parameters
max_proc=50

#Set Misc Experiment Parameters
algo="icp" #  "irm"
paramfile="$expdir/$algo_paramfile.pkl"

#Set Dataset Parameters
dtype="adult"  #adult, german
reduce_dsize=(-1)
binarize=1  #0, 1
seeds=(1000)  # 8079 52 147 256 784 990 587 304 888)

ft_combos=('1')
if [ $dtype == "adult" ]
then
    data="~/causal_discovery/data/adult.csv"
fi
if [ $dtype == "german" ]
then
    data="~/causal_discovery/data/germanCredit.csv"
fi

#Environment Parameters
eq_estrat=-1  #-1, #samples_wanted
envplur="single"  #all_combos, single

if [ $dtype == "adult" ]
then
    if [ $binarize == 0 ]
    then
        env_vars=("workclass" "native-country" "occupation" "marital-status" "relationship")
    fi
    if [ $binarize == 1 ]
    then
        env_vars=("workclass")   #"native-country" "occupation" "marital-status" "relationship")
    fi
fi

if [ $dtype == "german" ]
then
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
id=0
for red_d in ${reduce_dsize[*]}
do
  for s in ${seeds[*]}
  do
    for f_eng in ${ft_combos[*]}
    do
        python setup_params.py $id $algo $f_eng $data $expdir $cmdfile $paramfile ${env_vars[@]} -envcombos $envplur -reduce_dsize $red_d -binarize $binarize -eq_estrat $eq_estrat -seed $s
        id=$(($id + ${#env_vars[@]}))
    done
  done
done

#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

cmd=( $cmd )
num_tokens=${#cmd[@]}
xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile




#Note - must be run from the causal_discovery directory
