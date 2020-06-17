# #Functions
get_envs () {
  local dset_type=$1
  local bin_val=$2


  if [ $dset_type == "adult" ]
  then
      if [ $bin_val == 0 ]
      then
          env_vars=("workclass" "native-country" "relationship")  #"occupation" "marital-status" "relationship")
      fi
      if [ $bin_val == 1 ]
      then
          env_vars=("workclass" "native-country")   #"occupation" "marital-status" "relationship"
      fi
  fi

  if [ $dset_type == "german" ]
  then
      if [ $bin_val == 0 ]
      then
          env_vars=("Purpose" "Housing")  # 'OtherInstallmentPlans'  'Foreign' 'Savings' 'Personal' 'OtherDebtors')
      fi
      if [ $bin_val == 1 ]
      then
        env_vars=("Purpose" "Housing" "Property")
      fi
  fi
}

get_datapath () {
  local dset_type=$1
  if [ $dset_type == "adult" ]
  then
      echo "~/causal_discovery/data/adult.csv"
  fi
  if [ $dset_type == "german" ]
  then
      echo "~/causal_discovery/data/germanCredit.csv"
  fi
}

get_testset () {
  local dset_type=$1
  local algo=$2
  local vind=$3
  local env=$4

  if [ $vind == "0" ]
  then
      test_info="-1"

  elif [ $algo == "linreg"  -o  $algo == "logreg"  -o  $algo == "mlp" -o  $algo == "constant" ]
  then
      if [ $dset_type == "adult" ]
      then
          test_info=("workclass_DUMmY" "native-country_DUMmY" "relationship_DUMmY")
      elif [ $dset_type == "german" ]
      then
          test_info=("Purpose_DUMmY" "Housing_DUMmY")
      else
          echo Unimplemented Dset
          exit 42
      fi

  elif [ $algo == "icp"  -o  $algo == "irm"  -o  $algo == "linear-irm" ]
  then
      if [ $dset_type == "adult" ]
      then
          case $env in
            "workclass")
              test_info="workclass_DUMmY"  ;;
            "native-country")
              test_info="native-country_DUMmY"  ;;
            "occupation")
              test_info='occupation_DUMmY'  ;;
            "marital-status")
              test_info='marital-status_DUMmY'  ;;
            "relationship")
              test_info='relationship_DUMmY'  ;;
            *)
              echo unimplemented env
              exit 42
          esac
      elif [ $dset_type == "german" ]
      then
          echo 2
        case $env in
          "Purpose")
            test_info="Purpose_DUMmY"  ;;
          "Housing")
            test_info="Housing_DUMmY"  ;;
          "Property")
            test_info="Property_DUMmY"  ;;
          *)
            echo unimplemented env
            exit 42
        esac
      else
          echo Unimplemented Dset
          exit 42
      fi
  else
      echo Unimplemented Algorithm
      exit 42
  fi
}
