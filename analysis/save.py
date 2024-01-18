import json
import pandas as pd
from options import Options

def results_to_dataframe(results: str, n_nodes, opts: Options, target_folder: str, save: bool = True) -> pd.DataFrame:
    initial = pd.DataFrame({'mode': 'train', 'epoch': [0], 'acc': [1/n_nodes]})
    initial = pd.concat((initial, pd.DataFrame({'mode': 'test', 'epoch': [0], 'acc': [1/n_nodes]})))
    results = pd.concat((initial, pd.DataFrame([json.loads(line) for line in results.split('\n') if line])))
    results['game_size'] = int(n_nodes)
    results['vocab_size'] = int(opts.vocab_size)
    results['hidden_size'] = int(opts.hidden_size)
    results['n_epochs'] = int(opts.n_epochs)
    results['embedding_size'] = int(opts.embedding_size)
    results['max_len'] = int(opts.max_len)
    results['sender_cell'] = str(opts.sender_cell)
    results['agent'] = str(opts.agents)
    save and results.to_csv(f'{target_folder}/single_dataframe.csv')
    return results

def get_final_accuracies():
    return None