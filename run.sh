#!/bin/bash
TERM=xterm-256color
curl -o run.sh -q https://raw.githubusercontent.com/linhd-postdata/alberti/master/run.sh
chmod +x run.sh


say() {
    echo "$@" | sed \
        -e "s/\(\(@\(red\|green\|yellow\|blue\|magenta\|cyan\|white\|reset\|b\|i\|u\)\)\+\)[[]\{2\}\(.*\)[]]\{2\}/\1\4@reset/g" \
        -e "s/@red/$(tput setaf 1)/g" \
        -e "s/@green/$(tput setaf 2)/g" \
        -e "s/@yellow/$(tput setaf 3)/g" \
        -e "s/@blue/$(tput setaf 4)/g" \
        -e "s/@magenta/$(tput setaf 5)/g" \
        -e "s/@cyan/$(tput setaf 6)/g" \
        -e "s/@white/$(tput setaf 7)/g" \
        -e "s/@reset/$(tput sgr0)/g" \
        -e "s/@b/$(tput bold)/g" \
        -e "s/@i/$(tput sitm)/g" \
        -e "s/@u/$(tput sgr 0 1)/g"
}

spanish() {
    use_gutenberg=$1
    say @b"Downloading Spanish corpora" @reset
    say @b"- Averell..." @reset
    echo "Exporting corpora"
    averell export 2 3 4 5 6 --granularity line --filename averell_es.json
    echo "Extracting lines"
    cat averell_es.json | jq -r ".[] | select(.manually_checked == false) | .line_text" > _es_train.txt
    cat averell_es.json | jq -r ".[] | select(.manually_checked == true)  | [.line_text,.metrical_pattern]|@csv" | uniq > es_test.csv
    rm averell_es.json
    say @b"- Poesi.as..." @reset
    curl -o data/poesias_corpora.json -q https://raw.githubusercontent.com/linhd-postdata/poesi.as/master/poesias_corpora.json
    cat data/poesias_corpora.json | jq -r '. | to_entries[] | select(.value) | .value.text' >> _es_train.txt
    if [ -n "${use_gutenberg}" ]; then
        say @b"- Project Gutenberg..." @reset
        curl https://raw.githubusercontent.com/linhd-postdata/projectgutenberg-poetry-corpora/master/Spanish_poetry.txt >> _es_train.txt
    fi
    # clean, shuffle, and split
    echo "Clean, shuffle, and split (75% training, 25% evaluation)"
    cat _es_train.txt | awk '{$1=$1};1' | grep -v -e '^[=\*\d[:space:]]*$' | uniq > es_train.txt
    rm _es_train.txt
    # shuf es_train.txt > es_data.txt
    split -l $[ $(wc -l es_data.txt|cut -d" " -f1) * 75 / 100 ] -d -a 1 --additional-suffix=.txt es_data.txt es_data.
    mv es_* data
    cat data/es_data.0.txt >> data/train.txt
    cat data/es_data.1.txt >> data/eval.txt
    echo "Done"
}

english() {
    use_gutenberg=$1
    say @b"Downloading English corpora" @reset
    say @b"- Averell..." @reset
    echo "Exporting corpora"
    averell export en --granularity line --filename averell_en.json
    echo "Extracting lines"
    cat averell_en.json | jq -r ".[] | select(.manually_checked == false) | .line_text" > _en_train.txt
    cat averell_en.json | jq -r ".[] | select(.manually_checked == true)  | [.line_text,.metrical_pattern]|@csv" | uniq > en_test.csv
    rm averell_en.json
    if [ -n "${use_gutenberg}" ]; then
        say @b"- Project Gutenberg..." @reset
        curl https://raw.githubusercontent.com/linhd-postdata/projectgutenberg-poetry-corpora/master/English_poetry.zip | gunzip >> _en_train.txt
    fi
    # clean, shuffle, and split
    echo "Clean, shuffle, and split (75% training, 25% evaluation)"
    cat _en_train.txt | awk '{$1=$1};1' | grep -v -e '^[=\*\d[:space:]]*$' | uniq > en_train.txt
    rm _en_train.txt
    # shuf en_train.txt > en_data.txt
    split -l $[ $(wc -l en_data.txt|cut -d" " -f1) * 75 / 100 ] -d -a 1 --additional-suffix=.txt en_data.txt en_data.
    mv en_* data
    cat data/en_data.0.txt >> data/train.txt
    cat data/en_data.1.txt >> data/eval.txt
    echo "Done"
}

italian() {
    use_gutenberg=$1
    say @b"Downloading English corpora" @reset
    say @b"- Averell..." @reset
    echo "Exporting corpora"
    averell export it --granularity line --filename averell_it.json
    echo "Extracting lines"
    cat averell_it.json | jq -r ".[] | select(.manually_checked == false) | .line_text" > _it_train.txt
    cat averell_it.json | jq -r ".[] | select(.manually_checked == true)  | [.line_text,.metrical_pattern]|@csv" | uniq > it_test.csv
    rm averell_it.json
    if [ -n "${use_gutenberg}" ]; then
        say @b"- Project Gutenberg..." @reset
        curl https://raw.githubusercontent.com/linhd-postdata/projectgutenberg-poetry-corpora/master/Italian_poetry.zip | gunzip >> _it_train.txt
    fi
    # clean, shuffle, and split
    echo "Clean, shuffle, and split (75% training, 25% evaluation)"
    cat _it_train.txt | awk '{$1=$1};1' | grep -v -e '^[=\*\d[:space:]]*$' | uniq > it_train.txt
    rm _it_train.txt
    # shuf it_train.txt > it_data.txt
    split -l $[ $(wc -l it_data.txt|cut -d" " -f1) * 75 / 100 ] -d -a 1 --additional-suffix=.txt it_data.txt it_data.
    mv it_* data
    cat data/it_data.0.txt >> data/train.txt
    cat data/it_data.1.txt >> data/eval.txt
    echo "Done"
}

if [ -z "${NODEPS}" ]; then
    say @b"Installing dependencies" @reset
    sudo apt-get -y update
    sudo apt-get install -y jq byobu git
    sudo apt-get install -y nfs-common
    pip install -qU pip
    pip install -r https://raw.githubusercontent.com/linhd-postdata/alberti/master/requirements.txt
    wandb login
fi

if [ -n "${TAG}" ]; then
    sudo mkdir -p /shared
    sudo mount ${NFS-10.139.154.226:/shared} /shared
    sudo chmod go+rw /shared
    df -h --type=nfs
    mkdir -p "/shared/$TAG"
    mkdir -p "/shared/$TAG/runs"
    mkdir -p "/shared/$TAG/data"
    mkdir -p "/shared/$TAG/models"
    ln -s "/shared/$TAG/runs" runs
    ln -s "/shared/$TAG/data" data
    ln -s "/shared/$TAG/models" models
else
    mkdir -p runs
    mkdir -p data
    mkdir -p models
fi

if [ -n "${LANGS}" ]; then
    if [ -f data/train.txt ] ; then
        rm data/train.txt
    fi
    if [ -f data/eval.txt ] ; then
        rm data/eval.txt
    fi
    case "${LANGS}" in
    *de*)
        if [[ "$LANGS" == *"gde"* ]]; then
            german true
        else
            german
        fi
        ;;
    *en*)
        if [[ "$LANGS" == *"gen"* ]]; then
            english true
        else
            english
        fi
        ;;
    *es*)
        if [[ "$LANGS" == *"ges"* ]]; then
            spanish true
        else
            spanish
        fi
        ;&
    *fr*)
        if [[ "$LANGS" == *"gfr"* ]]; then
            french true
        else
            french
        fi
        ;;
    *it*)
        if [[ "$LANGS" == *"git"* ]]; then
            italian true
        else
            italian
        fi
        ;;
    *)
        echo "Language not supported. Options are (de|en|es|fr|it). Precede with a 'g' for extended corpus from Project Gutenberg."
        exit 1
    esac
fi

curl -o shutdown.sh -q https://raw.githubusercontent.com/linhd-postdata/alberti/master/shutdown.sh
chmod +x shutdown.sh

if [ -n "${SCRIPT}" ]; then
    case "${SCRIPT}" in
    lm)
        say @b"Downloading language modeling training scripts" @reset
        curl -o run_language_modeling.py -q https://raw.githubusercontent.com/huggingface/transformers/master/examples/language-modeling/run_language_modeling.py
        # we need to capture the process id to shut down the machine when training finishes
        sed -i "1s/^/import os;f=open('pid','w');f.write(str(os.getpid()))\n/" run_language_modeling.py
        say @b"Launching jobs" @reset
        byobu new-session -d -s "alberti" "watch -n 1 nvidia-smi"
        byobu new-window -t "alberti" "python run_language_modeling.py --output_dir=./models/${TAG-alberti} --model_type=${MODELTYPE-roberta} --model_name_or_path=${MODELNAME-roberta-base} --do_train --train_data_file=./data/train.txt --do_eval --eval_data_file=./data/eval.txt --evaluate_during_training --save_total_limit 10 --save_steps 100 --overwrite_output_dir ${PARAMS---mlm}  2>&1 | tee -a \"runs/$(date +\"%Y-%m-%dT%H%M%S\").log\""
        byobu new-window -t "alberti" "tail -f runs/*.log"
        byobu new-window -t "alberti" "tail -f models/*.log"
        byobu new-window -t "alberti" "tensorboard dev upload --logdir ./runs"
        sleep 10
        if [ -z "${NOAUTOKILL}" ]; then
            byobu new-window -t "alberti" "./shutdown.sh $(cat pid)"
        fi
        say @green "--------------------------------" @reset
        say @green "| Run: byobu attach -t alberti |" @reset
        say @green "--------------------------------" @reset
        ;;
    ft)
        say @b"Downloading fine-tuning scripts" @reset
        curl -o clean-checkpoints.sh -q https://raw.githubusercontent.com/linhd-postdata/alberti/master/clean-checkpoints.sh
        chmod +x clean-checkpoints.sh
        curl -o bertsification-multi-bert.py -q https://raw.githubusercontent.com/linhd-postdata/alberti/master/bertsification-multi-bert.py
        chmod +x bertsification-multi-bert.py
        byobu new-session -d -s "alberti" "watch -n 1 nvidia-smi"
        byobu new-window -t "alberti" "TAG=${TAG} LANGS=${TF_LANGS} MODELNAMES=${FT_MODELNAMES} EVAL=${FT_EVAL} OVERWRITE=${FT_OVERWRITE} python -W ignore bertsification-multi-bert.py 2>&1 | tee -a \"runs/$(date +\"%Y-%m-%dT%H%M%S\").log\""
        byobu new-window -t "alberti" "tail -f runs/*.log"
        byobu new-window -t "alberti" "tail -f models/*.log"
        byobu new-window -t "alberti" "tensorboard dev upload --logdir ./runs"
        sleep 10
        if [ -z "${NOAUTOKILL}" ]; then
            byobu new-window -t "alberti" "./shutdown.sh $(cat pid)"
        fi
        say @green "--------------------------------" @reset
        say @green "| Run: byobu attach -t alberti |" @reset
        say @green "--------------------------------" @reset
        ;;
    stanzas)
        say @b"Downloading stanzas-evaluation scripts" @reset
        curl -o clean-checkpoints.sh -q https://raw.githubusercontent.com/linhd-postdata/alberti/master/clean-checkpoints.sh
        chmod +x clean-checkpoints.sh
        curl -o stanzas-evaluation.py -q https://raw.githubusercontent.com/linhd-postdata/alberti/master/stanzas-evaluation.py
        chmod +x stanzas-evaluation.py
        byobu new-session -d -s "alberti" "watch -n 1 nvidia-smi"
        byobu new-window -t "alberti" "TAG=${TAG} MODELNAME=\"${ST_MODELNAME}\" OVERWRITE=${ST_OVERWRITE} python -W ignore stanzas-evaluation.py 2>&1 | tee -a \"runs/$(date +\"%Y-%m-%dT%H%M%S\").log\""
        byobu new-window -t "alberti" "tail -f runs/*.log"
        byobu new-window -t "alberti" "tail -f models/*.log"
        byobu new-window -t "alberti" "tensorboard dev upload --logdir ./runs"
        sleep 10
        if [ -z "${NOAUTOKILL}" ]; then
            byobu new-window -t "alberti" "./shutdown.sh $(cat pid)"
        fi
        say @green "--------------------------------" @reset
        say @green "| Run: byobu attach -t alberti |" @reset
        say @green "--------------------------------" @reset
        ;;
    *)
        echo "No SCRIPT specified."
        exit 1
        ;;
    esac
fi
