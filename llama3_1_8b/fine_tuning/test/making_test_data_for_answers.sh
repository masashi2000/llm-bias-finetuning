#!/bin/bash
# nohup ./process.sh > output2.log 2>&1 &

echo "Start"

python answer_generation/generate_answer_for_question.py --config_file ../config.yml --persona_file persona/dem_v3.txt --question_file question_generation/questions.csv

echo "Middle"
python answer_generation/generate_answer_for_question.py --config_file ../config.yml --persona_file persona/rep_v3.txt --question_file question_generation/questions.csv
