#!/bin/bash
. "$(pwd)/launchfiles/helper_funcs.sh"



#Make expieriment directories/files
expdir="/scratch/hdd001/home/adragnar/experiments/causal_discovery/0610_linearirm/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/0610_linearirm_hyp/$(date +'%s')"  #"/scratch/hdd001/home/adragnar/experiments/causal_discovery/$(date +'%s')"  #"/scratch/gobi1/adragnar/experiments/causal_discovery/$(date +'%s')"
mkdir -p $expdir
cmdfile="$expdir/cmdfile.sh"

#Set cluster parameters
max_proc=50

#Set Misc Experiment Parameters
use_test=1
algo="linear-irm" #  "icp" "linreg"
paramfile="$expdir/${algo}_paramfile.pkl"

#Set Dataset Parameters
dtypes=("adult" "german")
reduce_dsize=-1
bin=0  #0, 1
ft_combos='-1'   #'1' '12')
#Environment Parameters
eq_estrat=-1  #-1, #samples_wanted

seeds=(1000 8079)  # 1000 8079 52 147 256 784 990 587 304 888)
l_rates=(0.0001 0.001 .01)
l2_regs=(0)
n_anneals=(1)
n_iters=(1000 2000 5000)
penreg=(10000 5000 3000 1000) # 10000 20000)
hid_layers=(50 100 200) # 200)
val_split=0.10

#Generate the commandfile
id=0
  for s in ${seeds[*]}
  do
    for lr in ${l_rates[*]}
    do
      for l2 in ${l2_regs[*]}
      do
        for n_ann in ${n_anneals[*]}
        do
          for it in ${n_iters[*]}
          do
            for pw in ${penreg[*]}
            do
              for nh in ${hid_layers[*]}
              do
                for d in ${dtypes[*]}
                do
                  data=$(get_datapath $d)
                  if [ $d == "adult"  ]
                  then
                    env_info=("workclass" "native-country" "relationship")
                  else
                    env_info=("Purpose" "Housing")
                  fi
                  for e in ${env_info[*]}
                  do
                    get_testset $d 'linear-irm' 1 $e
                    python setup_params.py $id $algo $data $expdir $cmdfile $paramfile -env_att $e -fteng $ft_combos -reduce_dsize $reduce_dsize -binarize $bin -eq_estrat $eq_estrat -seed $s -test_info $test_info -inc_hyperparams 1 -irm_lr $lr -irm_niter $it -irm_l2 $l2 -irm_penalty_anneal $n_ann -irm_penalty_weight $pw -irm_hid_layers $nh -val_split $val_split
                    id=$(($id + 1))
                  done
                done
              done
            done
          done
        done
      done
    done
  done


#Run evaluation on cluster
num_cmds=`wc -l $cmdfile | cut -d' ' -f1`
echo "Wrote $num_cmds commands to $cmdfile"

cmd=( $cmd )
num_tokens=${#cmd[@]}
# xargs -L 1 -P $max_proc srun --mem=16G -p cpu < $cmdfile
