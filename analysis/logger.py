import torch
import egg.core as core

class ResultsCollector(core.Callback):
    def __init__(self, print_train_loss=True, compute_topsim_train_set=False, compute_topsim_test_set=True):
        self.print_train_loss = print_train_loss
        self.topsim_calculator = core.TopographicSimilarity(
            sender_input_distance_fn="edit", 
            message_distance_fn="edit", 
            compute_topsim_train_set=compute_topsim_train_set, 
            compute_topsim_test_set=compute_topsim_test_set, 
            is_gumbel=True
        )
        self.message_entropy_calculator = core.MessageEntropy(
            print_train=False,
            is_gumbel=True
        )
        self.results = []

    def on_epoch_end(self, loss: float, logs: core.Interaction, epoch: int):
        train_metrics = self._aggregate_metrics(loss, logs, "train", epoch)
        train_metrics["messages"] = self._messages_to_indices(logs.message)
        self.results.append(train_metrics)
        if self.print_train_loss:
            self._print_to_console({k: v for k, v in train_metrics.items() if k != 'messages'})

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        test_metrics = self._aggregate_metrics(loss, logs, "test", epoch)
        topsim = self.topsim_calculator.compute_topsim(
            torch.flatten(logs.sender_input, start_dim=1), 
            logs.message.argmax(dim=-1) if self.topsim_calculator.is_gumbel else logs.message
        )
        test_metrics["topsim"] = topsim
        test_metrics["messages"] = self._messages_to_indices(logs.message)
        test_metrics["message_entropy"] = self.message_entropy_calculator.print_message_entropy(logs, "test", epoch)
        self.results.append(test_metrics)
        self._print_to_console({k: v for k, v in test_metrics.items() if k != 'messages'})

    def _aggregate_metrics(self, loss: float, logs: core.Interaction, mode: str, epoch: int) -> dict:
        metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        return {
            "epoch": epoch,
            "mode": mode,
            "loss": loss,
            **metrics
        }
    
    def _messages_to_indices(self, messages_tensor):
        return [message.argmax(dim=-1).tolist() for message in messages_tensor]

    def _print_to_console(self, metrics: dict):
        output_message = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        print(output_message, flush=True)

    def get_results(self):
        return self.results