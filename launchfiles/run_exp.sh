#!/bin/bash
. "$(pwd)/launchfiles/helper_funcs.sh"



#Make expieriment directories/files
expdir="test_expdir"   #"/scratch/hdd001/home/adragnar/experiments/causal_discovery/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"

#Set cluster parameters
max_proc=50

#Set Misc Experiment Parameters
use_val=1
algo="icp" #  "icp" "linreg"
paramfile="$expdir/${algo}_paramfile.pkl"

#Set Dataset Parameters
dtypes=("adult" "german")
reduce_dsize=(-1)
bin=(0)  #0, 1
seeds=(1000)  # 1000 8079 52 147 256 784 990 587 304 888)
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
          if [ $algo == "icp"  -o  $algo == "irm" ]
          then
            get_envs $d $b   #Sets variable env_vars
              for e in ${env_vars[*]}
              do
                get_val $d $algo $use_val $e  #Sets variable val_info
                python setup_params.py $id $algo $data $expdir $cmdfile $paramfile -env_att $e -fteng $f_eng -reduce_dsize $red_d -binarize $b -eq_estrat $eq_estrat -seed $s -val $val_info
                id=$(($id + 1))
              done
            else
                get_val $d $algo $use_val  #Sets variable val_info
                python setup_params.py $id $algo $data $expdir $cmdfile $paramfile -fteng $f_eng -reduce_dsize $red_d -binarize $b -seed $s -val $val_info
                id=$(($id + 1))
            fi

        done
      done
    done
  done
done

#Run evaluation on cluster
# num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
# echo "Wrote $num_cmds commands to $cmdfile"
#
# cmd=( $cmd )
# num_tokens=${#cmd[@]}
# xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile




#Note - must be run from the causal_discovery directory
