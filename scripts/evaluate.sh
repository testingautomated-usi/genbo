#!/bin/bash

echo Running on python environment: $CONDA_DEFAULT_ENV

model=
num_run=
track_num=-1
port=-1
regression="false"
num_episodes=100
max_steps=1000
donkey_exe_path=

while [ $# -gt 0 ] ; do
  case $1 in
    -m | --model) model="$2" ;;
    -t | --track-num) track_num="$2" ;;
    -a | --port) port="$2" ;;
    -s | --max-steps) max_steps="$2" ;;
    -c | --num-run) num_run="$2" ;;
    -n | --num-episodes) num_episodes="$2" ;;
    -d | --donkey-exe-path) donkey_exe_path="$2" ;;
  esac
  shift
done

echo Model: "$model"
echo Track num: "$track_num"
echo Num run: "$num_run"
echo Add to port: "$port"
echo Max steps: "$max_steps"
echo Num episodes: "$num_episodes"
echo Donkey exe path: "$donkey_exe_path"

if test -z "$donkey_exe_path"; then
  echo "Donkey exe path is not set"
  exit 1
fi

if test -z "$model"; then

  if [[ "$track_num" < 0 ]]; then
    echo "Track not set"
    exit 1
  fi

  directory=$PWD
  cd ..

  echo "Collecting images on track $track_num using the autopilot agent"

  python evaluate.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
    --donkey-exe-path "$donkey_exe_path" --seed 0 --num-episodes "$num_episodes" \
    --agent-type autopilot --road-test-generator constant \
    --max-steps "$max_steps" --add-to-port "$port" --headless

  cd $directory
  exit 0
fi

model_without_extension="${model%.*}"

if [[ $model_without_extension == *"-run-"* ]]; then
  if test -z "$num_run"; then
    echo Num run needs to be provided
    exit 1
  fi
fi


directory=$PWD
cd ..

if [[ "$track_num" < 0 ]]; then

  echo Evaluating $model_without_extension on all tracks

  # try on all the tracks
  for i in 0 1 2 3 4 5 6 7 8; do

    echo Evaluating $model_without_extension on track "$i"

    if [[ $model_without_extension == *"-run-"* ]]; then
      log_name=logs/"$model_without_extension"-track-"$i"-run-"$num_run".txt
    else
      log_name=logs/"$model_without_extension"-track-"$i".txt
    fi
    
    python evaluate.py --env-name donkey --donkey-scene-name generated_track --track-num "$i" \
      --donkey-exe-path "$donkey_exe_path" --seed 0 --num-episodes "$num_episodes" \
      --agent-type supervised --model-path logs/models/"$model" --road-test-generator constant \
      --max-steps "$max_steps" --no-save-archive --add-to-port "$port" \
      --headless > $log_name

  done
else
  
  if [[ $model_without_extension == *"-run-"* ]]; then
    log_name=logs/"$model_without_extension"-track-"$track_num"-run-"$num_run".txt
  else
    log_name=logs/"$model_without_extension"-track-"$track_num".txt
  fi

  python evaluate.py --env-name donkey --donkey-scene-name generated_track --track-num "$track_num" \
    --donkey-exe-path "$donkey_exe_path" --seed 0 --num-episodes "$num_episodes" \
    --agent-type supervised --model-path logs/models/"$model" --road-test-generator constant \
    --max-steps "$max_steps" --no-save-archive --add-to-port "$port" \
    --headless > $log_name
fi

cd $directory

