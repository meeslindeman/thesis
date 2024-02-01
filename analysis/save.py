import json
import pandas as pd
from options import Options

def results_to_dataframe(results: list[dict], n_nodes, opts: Options, target_folder: str, save: bool = True) -> pd.DataFrame:
    for result in results:
        if 'messages' in result:
            result['messages'] = str(result['messages'])
            
    # Create initial DataFrame with baseline accuracy
    initial = pd.DataFrame({'mode': ['train', 'test'], 'epoch': [0, 0], 'acc': [1/n_nodes, 1/n_nodes]})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df = pd.concat([initial, results_df], ignore_index=True)

    # Add additional columns from opts
    results_df['game_size'] = int(n_nodes)
    results_df['vocab_size'] = int(opts.vocab_size)
    results_df['hidden_size'] = int(opts.hidden_size)
    results_df['n_epochs'] = int(opts.n_epochs)
    results_df['embedding_size'] = int(opts.embedding_size)
    results_df['heads'] = int(opts.heads)
    results_df['max_len'] = int(opts.max_len)
    results_df['sender_cell'] = str(opts.sender_cell)
    results_df['agent'] = str(opts.agents)
    results_df['batch_size'] = int(opts.batch_size)
    results_df['random_seed'] = int(opts.random_seed)

    if save:
        results_df.to_csv(f'{target_folder}/single_dataframe.csv', index=False)

    return results_df