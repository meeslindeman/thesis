import json
import egg.core as core
from options import Options

class ResultsCollector(core.ConsoleLogger):
    def __init__(self, results: list, print_to_console: bool, print_train_loss=True, as_json=True, callbacks_list=None):
        super().__init__(as_json=as_json, print_train_loss=print_train_loss)
        self.results = results
        self.print_to_console = print_to_console
        self.callbacks_list = callbacks_list

    def aggregate_print(self, loss: float, logs, mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())

        if self.callbacks_list:
            for callback in self.callbacks_list:
                if isinstance(callback, core.TopographicSimilarity):
                    topsim = callback.print_message(logs, mode, epoch)
                    dump.update(dict(topsim=topsim))

        dump.update(aggregated_metrics)
        dump.update(dict(mode=mode, epoch=epoch))

        results = json.dumps(dump)
        self.results.append(results)

        if self.print_to_console:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
            print(output_message)

def get_callbacks(opts):
    callbacks = []
    if opts.callbacks_config['early_stopper']:
        callbacks.append(core.EarlyStopperAccuracy(opts.accuracy, validation=False))
    if opts.callbacks_config['message_entropy']:
        callbacks.append(core.MessageEntropy(print_train=True, is_gumbel=True))
    if opts.callbacks_config['print_validation']:
        callbacks.append(core.PrintValidationEvents(n_epochs=opts.n_epochs))
    if opts.callbacks_config['topographic_similarity']:
        callbacks.append(core.TopographicSimilarity(
            sender_input_distance_fn="hamming", 
            message_distance_fn="euclidean", 
            compute_topsim_train_set=False, 
            compute_topsim_test_set=True, 
            is_gumbel=True
        ))
    if opts.callbacks_config['disent']:
        callbacks.append(core.Disent(
            is_gumbel=True, 
            compute_posdis=True, 
            compute_bosdis=False, 
            vocab_size=opts.vocab_size,
            print_test=True,
            print_train=True
        ))
    return callbacks