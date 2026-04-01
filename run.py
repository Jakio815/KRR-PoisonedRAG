import os

def run(test_params):

    log_file, log_name = get_log_name(test_params)

    cmd = f"python main.py " \
        f"--eval_model_code {test_params['eval_model_code']} " \
        f"--eval_dataset {test_params['eval_dataset']} " \
        f"--split {test_params['split']} " \
        f"--query_results_dir {test_params['query_results_dir']} " \
        f"--model_name {test_params['model_name']} " \
        f"--top_k {test_params['top_k']} " \
        f"--use_truth {test_params['use_truth']} " \
        f"--gpu_id {test_params['gpu_id']} " \
        f"--attack_method {test_params['attack_method']} " \
        f"--adv_per_query {test_params['adv_per_query']} " \
        f"--score_function {test_params['score_function']} " \
        f"--repeat_times {test_params['repeat_times']} " \
        f"--M {test_params['M']} " \
        f"--seed {test_params['seed']} " \
        f"--name {log_name} " \
        f"> {log_file}"
        
    os.system(cmd)


def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"logs/{test_params['query_results_dir']}_logs", exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"

    if test_params['note'] != None:
        log_name = test_params['note']
    
    return f"logs/{test_params['query_results_dir']}_logs/{log_name}.txt", log_name



test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",
    'split': "test",
    'query_results_dir': 'main',

    # LLM setting
    'model_name': 'palm2', 
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 0,

    # attack
    'attack_method': 'prompt_injection',
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times': 1,
    'M': 1,
    'seed': 12,

    'note': None
}

for dataset in ['nq', 'hotpotqa', 'msmarco']:
    test_params['eval_dataset'] = dataset
    run(test_params)