#!/bin/bash
echo "Installing dependencies"
sudo apt-get -y update
sudo apt-get install -y jq byobu
pip install -qU pip
pip install -r requirements.txt
if [ -z "${TAG}" ]; then
    sudo apt-get install -y nfs-common
    sudo mkdir -p /share
    sudo chmod go+rw /share
    sudo mount 10.225.255.2:/share /share
    df -h --type=nfs
    mkdir -p "/share/$TAG"
    mkdir -p "/share/$TAG/runs"
    mkdir -p "/share/$TAG/data"
    mkdir -p "/share/$TAG/models"
    ln -s "/share/$TAG/runs" runs
    ln -s "/share/$TAG/data" data
    ln -s "/share/$TAG/models" models
else
    mkdir data
    mkdir data
    mkdir models
fi

echo "Downloading corpora"
echo "- Averell..."
averell download 2 3 4 5 6
averell download 2 3 4 5 6
averell export 2 3 4 5 6 --granularity line
cat corpora/line.json | jq -r ".[] | select(.manually_checked == false) | .line_text" > _es_train.txt
cat corpora/line.json | jq -r ".[] | select(.manually_checked == true)  | [.line_text,.metrical_pattern]|@csv" | uniq > es_test.csv
rm corpora/line.json

echo "- Poesi.as..."
curl -o corpora/poesias_corpora.json -q https://raw.githubusercontent.com/linhd-postdata/poesi.as/master/poesias_corpora.json
cat corpora/poesias_corpora.json | jq -r '. | to_entries[] | select(.value) | .value.text' >> _es_train.txt
cat _es_train.txt | awk '{$1=$1};1' | grep -v -e '^[=\*\d[:space:]]*$' | uniq > es_train.txt
rm _es_train.txt
mv es_* data

echo "Downloading scripts"
curl -o run_language_modeling.py -q https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/run_language_modeling.py
sed -i "1s/^/import os;f=open('pid','w');f.write(str(os.getpid()))\n/" run_language_modelin
curl -o shutdown.sh -q https://raw.githubusercontent.com/linhd-postdata/alberti/master/shutdown.sh
chmod +x shutdown.sh

echo "Launching jobs"
byobu new-session -d -s "alberti" "watch -n 1 nvidia-smi"
byobu new-window -t "alberti" "python run_language_modeling.py --output_dir=./models/alBERTi-tiny-v2 --model_type=roberta --model_name_or_path=roberta-base --do_train --train_data_file=./data/train.txt --do_eval --eval_data_file=./data/eval.txt --evaluate_during_training --save_total_limit 5 --save_steps 1000 --mlm --overwrite_output_dir --n_gpu 2"
byobu new-window -t "alberti" "htop"
byobu new-window -t "alberti" "tensorboard dev upload --logdir ./runs"
byobu new-window -t "alberti" "./shutdown.sh $(cat pid)"
