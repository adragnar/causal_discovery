#!/bin/bash
. "$(pwd)/launchfiles/helper_funcs.sh"

#Make expieriment directories/files
expdir="/scratch/gobi1/adragnar/experiments/causal_discovery/0610_realreg_hyp/$(date +'%s')"  #"/scratch/hdd001/home/adragnar/experiments/causal_discovery/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"

#Set cluster parameters
max_proc=50

#Set Misc Experiment Parameters
use_test=1
val_split=0.20
algo="logreg" #  "icp" "linreg"
paramfile="$expdir/${algo}_paramfile.pkl"

#Set Dataset Parameters
dtypes=("adult" "german")  # "german"
reduce_dsize=-1
bin=0  #0, 1
ft_combos='-1'   #'1' '12')
#Environment Parameters
eq_estrat=-1  #-1, #samples_wanted

if [ $algo == "linreg"  ]
then
  hp_name='-linreg_lambda'
elif [ $algo == "logreg"  ]
then
  hp_name='-logreg_c'
fi
seeds=(1000 8079)  # 1000 8079 52 147 256 784 990 587 304 888)
lambdas=(0.001 0.0001 0.00001 0.000001)

#Generate the commandfile
id=0
for d in ${dtypes[*]}
do
  data=$(get_datapath $d)
  if [ $d == "adult"  ]
  then
    test_info=("workclass_DUMmY" "native-country_DUMmY" "relationship_DUMmY")
  else
    test_info=("Purpose_DUMmY" "Housing_DUMmY")
  fi


  for s in ${seeds[*]}
  do
    for t in ${test_info[*]}
    do
      for l in ${lambdas[*]}
      do

        python setup_params.py $id $algo $data $expdir $cmdfile $paramfile -fteng $ft_combos -reduce_dsize $reduce_dsize -binarize $bin -inc_hyperparams 1 -seed $s -test_info $t -val_split $val_split $hp_name $l
        id=$(($id + 1))

      done
    done
  done
done

#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

cmd=( $cmd )
num_tokens=${#cmd[@]}
xargs -L 1 -P $max_proc srun --mem=16G -p cpu <  $cmdfile
