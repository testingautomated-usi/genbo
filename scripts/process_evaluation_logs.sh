#!/bin/bash

model=
original="false"

while [ $# -gt 0 ] ; do
  case $1 in
    -m | --model) model="$2" ;;
    -o | --original) original="$2" ;;
  esac
  shift
done

if test -z "$model"; then
   echo Please provide the model name suffix
   echo The name of the file should be the name of the model, i.e., 'donkey-dave2-m4', without extension
   exit 1
fi

directory=$PWD

cd ../logs

if [[ "$original" == "true" ]]; then
    for i in $(ls | grep "$model"-track); do 
        echo $i >> results_model.txt 
        cat $i | grep -i "success rate" >> results_model.txt
    done 
    echo "------------ $model -----------"
    cat results_model.txt | grep -i "success rate"
    rm results_model.txt
else
    for j in 1 2 3 4 5 6 7 8 9; do 
        for i in $(ls | grep "$model"-run-$j); do 
            echo $i >> results_model_run$j.txt 
            cat $i | grep -i "success rate" >> results_model_run$j.txt
        done 
    done
    for i in 1 2 3 4 5 6 7 8 9; do 
        if test -f results_model_run$i.txt; then
          echo "------------ "$model"-run-$i -----------" 
          cat results_model_run$i.txt | grep -i "success rate"
          rm results_model_run$i.txt
        fi
    done
fi

cd $directory





