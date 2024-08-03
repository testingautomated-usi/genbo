#!/bin/bash

echo Running on python environment: $CONDA_DEFAULT_ENV

port=-1
num_iterations=10
num_restarts=40
num_runs_failure=3
seed_generator=performance
individual_generator=sequence
length_exponential_factor=3
lam=1
bias=true
track_num=0
model=
donkey_exe_path=
seed=-1

while [ $# -gt 0 ] ; do
  case $1 in
    -p | --port) port="$2" ;;
    -i | --num-iterations) num_iterations="$2" ;;
    -r | --num-restarts) num_restarts="$2" ;;
    -f | --num-runs-failure) num_runs_failure="$2" ;;
    -g | --individual-generator) individual_generator="$2" ;;
    -l | --length-exponential-factor) length_exponential_factor="$2" ;;
    -a | --lam) lam="$2" ;;
    -m | --model) model="$2" ;;
    -e | --donkey-exe-path) donkey_exe_path="$2" ;;
    -s | --seed) seed="$2" ;;
  esac
  shift
done

echo Port: "$port"
echo Num iterations: "$num_iterations"
echo Num restarts: "$num_restarts"
echo Num runs failure: "$num_runs_failure"
echo Individual generator: "$individual_generator"
echo Length exponential factor: "$length_exponential_factor"
echo Model: "$model"
echo Donkey exe path: "$donkey_exe_path"

if test -z "$donkey_exe_path"; then
  echo "Donkey exe path is not set"
  exit 1
fi

directory=$PWD
cd ..

if test -z "$model"; then
   # test autopilot
   python search.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
      --donkey-exe-path "$donkey_exe_path" \
      --agent-type autopilot --evaluator-name real --headless \
      --individual-name state_pair_individual --fitness-name cte_fitness \
      --seed-state-generator-name "$seed_generator" --individual-generator-name "$individual_generator" \
      --length-exponential-factor "$length_exponential_factor" --num-iterations "$num_iterations" \
      --num-restarts "$num_restarts" --lam $lam --num-runs-failure "$num_runs_failure" \
      --logging-level info --add-to-port "$port" --bias --mutate-both-members --num-runs 10 --seed "$seed"
else
   python search.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
      --donkey-exe-path "$donkey_exe_path" \
      --agent-type supervised --model-path logs/models/"$model" --evaluator-name real --headless \
      --individual-name state_pair_individual --fitness-name cte_fitness \
      --seed-state-generator-name "$seed_generator" --individual-generator-name "$individual_generator" \
      --length-exponential-factor "$length_exponential_factor" --num-iterations "$num_iterations" \
      --num-restarts "$num_restarts" --lam $lam --num-runs-failure "$num_runs_failure" \
      --logging-level info --add-to-port "$port" --bias --mutate-both-members --num-runs 10 --seed "$seed"
fi

cd "$directory"
