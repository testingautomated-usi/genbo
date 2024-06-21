#!/bin/bash

echo Running on python environment: $CONDA_DEFAULT_ENV

model_to_test=autopilot
individual_generator_name=sequence
model_tested=autopilot
exp_factor=3
track_num=0
headless=true
video=false
execute_failure_member=false
num_runs=5
port=-1
collect_images=false
max_steps=250
donkey_exe_path=
num_iterations=10
num_restarts=40

while [ $# -gt 0 ] ; do
  case $1 in
    -m | --model-to-test) model_to_test="$2" ;;
    -i | --individual-generator-name) individual_generator_name="$2" ;;
    -o | --model-tested) model_tested="$2" ;;
    -e | --exp-factor) exp_factor="$2" ;;
    -h | --headless) headless="$2" ;;
    -v | --video) video="$2" ;;
    -n | --num-runs) num_runs="$2" ;;
    -a | --port) port="$2" ;;
    -c | --collect-images) collect_images="$2" ;;
    -d | --donkey-exe-path) donkey_exe_path="$2" ;;
    -t | --num-iterations) num_iterations="$2" ;;
    -r | --num-restarts) num_restarts="$2" ;;
    -f | --execute-failure-member) execute_failure_member="$2" ;;
  esac
  shift
done

echo Model to test path: "$model_to_test"
echo Individual generator name: "$individual_generator_name"
echo Model tested: "$model_tested"
echo Exponential factor: "$exp_factor"
echo Execute failure member: "$execute_failure_member"
echo Video: "$video"
echo Num runs: "$num_runs"
echo Add to port: "$port"
echo Collect images: "$collect_images"
echo Max steps: "$max_steps"
echo Donkey exe path: "$donkey_exe_path"

if test -z "$donkey_exe_path"; then
  echo "Donkey exe path is not set"
  exit 1
fi

directory=$PWD
cd ..

# remove the extension
model_tested=$(echo "$model_tested" | cut -d '.' -f 1)
# remove donkey
model_tested=$(echo "$model_tested" | cut -d '-' -f2-)

if [[ "$model_tested" != "autopilot" ]]; then
  model_tested="supervised_$model_tested"
fi

if [[ "$individual_generator_name" == "sequence" ]]; then
  archive_names=$(find ./logs/test_generation/"$individual_generator_name"/state/donkey/generated_track_"$track_num" -name "*_agent_$model_tested*_exp_factor_$exp_factor.json" | cut -d '/' -f 8 | cut -d '.' -f 1)
elif [[ "$individual_generator_name" == "one_plus_lambda" ]]; then
  archive_names=$(find ./logs/test_generation/"$individual_generator_name"/state/donkey/generated_track_"$track_num" -name "*_agent_$model_tested*$num_iterations*$num_restarts.json" | cut -d '/' -f 8 | cut -d '.' -f 1)
else
  echo Uknown individual generator name "$individual_generator_name"
fi

if [[ -z "$archive_names" ]]; then
  echo Not possible to find a match for logs/test_generation/"$individual_generator_name"/state/donkey/generated_track_"$track_num"/*_agent_"$model_tested"*_exp_factor_"$exp_factor".json
  cd scripts || exit
  exit 1
fi

# No quotes on archive_names, which it is supposed to contain spaces

if [[ "$model_to_test" == "autopilot" ]]; then
  if [[ "$video" == "true" ]]; then
    if [[ "$execute_failure_member" == "true" ]]; then
      python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
        --donkey-exe-path "$donkey_exe_path" --agent-type autopilot --evaluator-name real \
        --individual-name state_pair_individual --individual-generator-name "$individual_generator_name" \
        --fitness-name cte_fitness --archive-filenames $archive_names \
        --num-runs "$num_runs" --mode recovery --logging-level info --headless \
        --video --execute-failure-member --add-to-port "$port"
    else
      python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
        --donkey-exe-path "$donkey_exe_path" --agent-type autopilot --evaluator-name real \
        --individual-name state_pair_individual --individual-generator-name "$individual_generator_name" \
        --fitness-name cte_fitness --archive-filenames $archive_names \
        --num-runs "$num_runs" --mode recovery --logging-level info --headless \
        --video --add-to-port "$port"
    fi
  else
    if [[ "$collect_images" == "true" ]]; then
      if [[ "$execute_failure_member" == "true" ]]; then
        python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
          --donkey-exe-path "$donkey_exe_path" --agent-type autopilot --evaluator-name real \
          --individual-name state_pair_individual --individual-generator-name "$individual_generator_name" \
          --fitness-name cte_fitness --archive-filenames $archive_names \
          --num-runs "$num_runs" --mode recovery --logging-level info --headless \
          --execute-failure-member --add-to-port "$port" --collect-images --max-steps "$max_steps"
      else 
        python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
          --donkey-exe-path "$donkey_exe_path" --agent-type autopilot --evaluator-name real \
          --individual-name state_pair_individual --individual-generator-name "$individual_generator_name" \
          --fitness-name cte_fitness --archive-filenames $archive_names \
          --num-runs "$num_runs" --mode recovery --logging-level info --headless \
          --execute-failure-member --add-to-port "$port" --collect-images --max-steps "$max_steps"
      fi
    else
      if [[ "$execute_failure_member" == "true" ]]; then
        python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
          --donkey-exe-path "$donkey_exe_path" --agent-type autopilot --evaluator-name real \
          --individual-name state_pair_individual --individual-generator-name "$individual_generator_name" \
          --fitness-name cte_fitness --archive-filenames $archive_names \
          --num-runs "$num_runs" --mode recovery --logging-level info --headless \
          --execute-failure-member --add-to-port "$port"
      else
        python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
          --donkey-exe-path "$donkey_exe_path" --agent-type autopilot --evaluator-name real \
          --individual-name state_pair_individual --individual-generator-name "$individual_generator_name" \
          --fitness-name cte_fitness --archive-filenames $archive_names \
          --num-runs "$num_runs" --mode recovery --logging-level info --headless \
          --add-to-port "$port"
      fi
    fi
  fi
else
  if [[ "$video" == "true" ]]; then
    if [[ "$execute_failure_member" == "true" ]]; then
      python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
        --donkey-exe-path "$donkey_exe_path" --agent-type supervised \
        --evaluator-name real --model-path logs/models/"$model_to_test" --individual-name state_pair_individual \
        --individual-generator-name "$individual_generator_name" \
        --fitness-name cte_fitness --archive-filenames $archive_names \
        --num-runs "$num_runs" --mode recovery --logging-level info --headless --video --execute-failure-member \
        --add-to-port "$port"
    else
      python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
        --donkey-exe-path "$donkey_exe_path" --agent-type supervised \
        --evaluator-name real --model-path logs/models/"$model_to_test" --individual-name state_pair_individual \
        --individual-generator-name "$individual_generator_name" \
        --fitness-name cte_fitness --archive-filenames $archive_names \
        --num-runs "$num_runs" --mode recovery --logging-level info --headless --video \
        --add-to-port "$port"
    fi
  else
    if [[ "$collect_images" == "true" ]]; then
      if [[ "$execute_failure_member" == "true" ]]; then
        python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
          --donkey-exe-path "$donkey_exe_path" --agent-type supervised \
          --evaluator-name real --model-path logs/models/"$model_to_test" --individual-name state_pair_individual \
          --individual-generator-name "$individual_generator_name" \
          --fitness-name cte_fitness --archive-filenames $archive_names \
          --num-runs "$num_runs" --mode recovery --logging-level info --headless --execute-failure-member \
          --add-to-port "$port" --collect-images --max-steps "$max_steps"
      else
        python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
          --donkey-exe-path "$donkey_exe_path" --agent-type supervised \
          --evaluator-name real --model-path logs/models/"$model_to_test" --individual-name state_pair_individual \
          --individual-generator-name "$individual_generator_name" \
          --fitness-name cte_fitness --archive-filenames $archive_names \
          --num-runs "$num_runs" --mode recovery --logging-level info --headless \
          --add-to-port "$port" --collect-images --max-steps "$max_steps"
      fi
    else
      if [[ "$execute_failure_member" == "true" ]]; then
        python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
          --donkey-exe-path "$donkey_exe_path" --agent-type supervised \
          --evaluator-name real --model-path logs/models/"$model_to_test" --individual-name state_pair_individual \
          --individual-generator-name "$individual_generator_name" \
          --fitness-name cte_fitness --archive-filenames $archive_names \
          --num-runs "$num_runs" --mode recovery --logging-level info --headless --execute-failure-member \
          --add-to-port "$port"
      else
        python replay.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
          --donkey-exe-path "$donkey_exe_path" --agent-type supervised \
          --evaluator-name real --model-path logs/models/"$model_to_test" --individual-name state_pair_individual \
          --individual-generator-name "$individual_generator_name" \
          --fitness-name cte_fitness --archive-filenames $archive_names \
          --num-runs "$num_runs" --mode recovery --logging-level info --headless \
          --add-to-port "$port"
      fi
    fi
  fi
fi

cd $directory

