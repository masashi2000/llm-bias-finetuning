#!/bin/bash
# nohup ./process.sh > output2.log 2>&1 &


python generate_response_for_answer.py --config_file ../../config.yml --question_and_answer_file ../answer_generation/questions_dem_v3.csv --persona_file ../persona/dem_v3.txt --template_file respond_prompt_template_v2.txt --name_file names.csv

python generate_response_for_answer.py --config_file ../../config.yml --question_and_answer_file ../answer_generation/questions_dem_v3.csv --persona_file ../persona/rep_v3.txt --template_file respond_prompt_template_v2.txt --name_file names.csv

python generate_response_for_answer.py --config_file ../../config.yml --question_and_answer_file ../answer_generation/questions_rep_v3.csv --persona_file ../persona/rep_v3.txt --template_file respond_prompt_template_v2.txt --name_file names.csv

python generate_response_for_answer.py --config_file ../../config.yml --question_and_answer_file ../answer_generation/questions_rep_v3.csv --persona_file ../persona/dem_v3.txt --template_file respond_prompt_template_v2.txt --name_file names.csv
