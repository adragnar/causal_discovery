#!/bin/bash
. "$(pwd)/launchfiles/helper_funcs.sh"



#Make expieriment directories/files
expdir="test_expdir" #"/scratch/hdd001/home/adragnar/experiments/causal_discovery/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/0610_logreg_test/$(date +'%s')"   #"/scratch/hdd001/home/adragnar/experiments/causal_discovery/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"

#Set cluster parameters
max_proc=50

#Set Misc Experiment Parameters
use_test=1
algo="constant" # "logreg" "icp" "linreg"
paramfile="$expdir/${algo}_paramfile.pkl"

#Set Dataset Parameters
dtypes=("adult" "german")  #"german"
reduce_dsize=(-1)
bin=(0)  #0, 1
seeds=(52)  # 1000 8079 52 147 256 784 990 587 304 888)
ft_combos=('-1')   #'1' '12')
#Environment Parameters
eq_estrat=-1  #-1, #samples_wanted

#Generate the commandfile
id=0
for red_d in ${reduce_dsize[*]}
do
  for s in ${seeds[*]}
  do
    for f_eng in ${ft_combos[*]}
    do
      for b in ${bin[*]}
      do
        for d in ${dtypes[*]}
        do

          data=$(get_datapath $d)
          if [ $algo == "icp"  -o  $algo == "irm"  -o  $algo == "linear-irm" ]
          then
            get_envs $d $b   #Sets variable env_vars
              for e in ${env_vars[*]}
              do
                get_testset $d $algo $use_test $e  #Sets variable val_info
                python setup_params.py $id $algo $data $expdir $cmdfile $paramfile -env_att $e -fteng $f_eng -reduce_dsize $red_d -binarize $b -eq_estrat $eq_estrat -seed $s -test_info $test_info
                id=$(($id + 1))
              done
          elif [ $algo == "linreg"  -o  $algo == "logreg"  -o  $algo == "mlp" -o  $algo == "constant" ]
          then
            get_testset $d $algo $use_test  #Sets variable val_info
              for t in ${test_info[*]}
              do
                python setup_params.py $id $algo $data $expdir $cmdfile $paramfile -fteng $f_eng -reduce_dsize $red_d -binarize $b -seed $s -test_info $t
                id=$(($id + 1))
              done
          fi

        done
      done
    done
  done
done

#Save code for reproducibility
python reproducibility.py $(pwd) $expdir

#Run evaluation on cluster
if [ $algo != "constant" ]
then
  num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
  echo "Wrote $num_cmds commands to $cmdfile"

  cmd=( $cmd )
  num_tokens=${#cmd[@]}
  xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile
fi



#Note - must be run from the causal_discovery directory
