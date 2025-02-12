#!/bin/bash

# First, generate all configurations and save them
python3 - << 'END_PYTHON'
import json
from itertools import product

COMPONENTS = {
    'response_formats': ['number_list', 'markdown'],
    'response_types': ['answer_first', 'fact_first'],
    'prompt_types': ['few_shot', 'zero_shot'],
    'explanation_types': ['structured', 'unstructured'],
    'seeds': ['seed_42', 'seed_3991'],
    'datasets': ['all_domains_75_samples', 'all_domains_all_samples'],
    'generation': ['greedy', 'temp_025', 'temp_06', 'temp_09'],
    'evaluation_datasets': ['test_set_1', 'test_set_2'],
    'quantisation': ['full_model']
}

# Generate eval configs
eval_configs = [
    f"{rf}_{rt}_{pt}_{et}"
    for rf, rt, pt, et in product(
        COMPONENTS['response_formats'],
        COMPONENTS['response_types'],
        COMPONENTS['prompt_types'],
        COMPONENTS['explanation_types']
    )
]

# Generate all configurations
configs = [
    f"model=llama3 tokenizer=llama3 seeds={seed} dataset={dataset} generation={gen} evaluation_dataset={eval_set} eval={eval_cfg} ++eval.quantisation_status={quant} ++eval.training_status=trained"
    for seed, dataset, gen, eval_set, quant, eval_cfg in product(
        COMPONENTS['seeds'],
        COMPONENTS['datasets'],
        COMPONENTS['generation'],
        COMPONENTS['evaluation_datasets'],
        COMPONENTS['quantisation'],
        eval_configs
    )
]

# Calculate number of chain links needed (95 jobs per link)
total_configs = len(configs)
chain_links = (total_configs + 94) // 95

print(f"Total configurations: {total_configs}")
print(f"Number of chain links needed: {chain_links}")

# Save all configurations
with open('/home/fr/fr_fr/fr_rf1031/bar-llama/helix_configs/all_configs.json', 'w') as f:
    json.dump(configs, f, indent=2)
END_PYTHON

# Get the number of chain links from the total number of configurations
chain_links=$(python3 -c "
import json
with open('/home/fr/fr_fr/fr_rf1031/bar-llama/helix_configs/all_configs.json') as f:
    configs = json.load(f)
print((len(configs) + 94) // 95)
")

# Define the chain job script
chain_link_job=${PWD}/chain_job.sh
dep_type="afterany"

myloop_counter=1
# Submit loop
while [ ${myloop_counter} -le ${chain_links} ] ; do
   if [ ${myloop_counter} -eq 1 ] ; then
      slurm_opt=""
   else
      slurm_opt="-d ${dep_type}:${jobID}"
   fi
   
   echo "Chain job iteration = ${myloop_counter}"
   echo "   sbatch --export=myloop_counter=${myloop_counter} ${slurm_opt} ${chain_link_job}"
   
   jobID=$(sbatch --export=ALL,myloop_counter=${myloop_counter} ${slurm_opt} ${chain_link_job} 2>&1 | sed 's/[S,a-z]* //g')
   
   if [[ "${jobID}" =~ "ERROR" ]] ; then
      echo "   -> submission failed!" ; exit 1
   else
      echo "   -> job number = ${jobID}"
   fi
   
   let myloop_counter+=1
done
