#!/bin/bash

#Make expieriment directories/files
expdir="/scratch/hdd001/home/adragnar/experiments/causal_discovery/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"

#Set cluster parameters
max_proc=50

#Set Misc Experiment Parameters
use_val=1
algo="icp" #  "icp" "linreg"
paramfile="$expdir/${algo}_paramfile.pkl"

#Set Dataset Parameters
dtype="adult"  #adult, german
reduce_dsize=(-1)
binarize=0  #0, 1
seeds=(1000)  # 1000 8079 52 147 256 784 990 587 304 888)

ft_combos=('-1')   #'1' '12')
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
        length_envvars=5
        env_vars="[workclass,native-country,occupation,marital-status,relationship]"
        val_info="[workclass_DUMmY,native-country_DUMmY,occupation_DUMmY,marital-status_DUMmY,relationship_DUMmY]"
    fi
    if [ $binarize == 1 ]
    then
        length_envvars=2
        env_vars="[workclass,native-country]"   #"occupation" "marital-status" "relationship")
        val_info="[workclass_DUMmY,native-country_DUMmY]"
    fi
fi

if [ $dtype == "german" ]
then
    if [ $binarize == 0 ]
    then
        length_envvars=4
        env_vars="[Purpose,Housing,Telephone,Property]"  # 'OtherInstallmentPlans'  'Foreign' 'Savings' 'Personal' 'OtherDebtors')
        val_info="[Purpose_DUMmY,Housing_DUMmY,Telephone_DUMmY,Property_DUMmY]"
    fi
    if [ $binarize == 1 ]
    then
       length_envvars=4
        env_vars="[Purpose,Housing,Telephone,Property]"
        val_info="[Purpose_DUMmY,Housing_DUMmY,Telephone_DUMmY,Property_DUMmY]"
    fi
fi

if [ $algo == "linreg" ]
then
    val_info="[Purpose_DUMmY]"
fi

#Set up validation info
if [ $use_val == 0 ]
then
    val_info="[-1]"
fi


#Generate the commandfile
id=0
for red_d in ${reduce_dsize[*]}
do
  for s in ${seeds[*]}
  do
    for f_eng in ${ft_combos[*]}
    do
        if [ $algo == "icp"  -o  $algo == "irm" ]
        then
            python setup_params.py $id $algo $data $expdir $cmdfile $paramfile -env_list $env_vars -envcombos $envplur -fteng $f_eng -reduce_dsize $red_d -binarize $binarize -eq_estrat $eq_estrat -seed $s -val $val_info
            id=$(($id + $length_envvars))
        else
            python setup_params.py $id $algo $data $expdir $cmdfile $paramfile -fteng $f_eng -reduce_dsize $red_d -binarize $binarize -seed $s -val $val_info
            id=$(($id + 1))
        fi
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
