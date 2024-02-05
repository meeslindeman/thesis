import torch
import egg.core as core

class ResultsCollector(core.Callback):
    """
    A class that collects and stores results during training and validation.

    Attributes:
        print_train_loss (bool): Whether to print the training loss.
        compute_topsim_train_set (bool): Whether to compute topographic similarity for the training set.
        compute_topsim_test_set (bool): Whether to compute topographic similarity for the test set.
        topsim_calculator (core.TopographicSimilarity): An instance of the TopographicSimilarity class.
        message_entropy_calculator (core.MessageEntropy): An instance of the MessageEntropy class.
        results (list): A list to store the collected results.

    Methods:
        on_epoch_end(loss, logs, epoch): Called at the end of each epoch during training.
        on_validation_end(loss, logs, epoch): Called at the end of each validation step.
        _aggregate_metrics(loss, logs, mode, epoch): Aggregates the metrics for a given mode (train or test).
        _messages_to_indices(messages_tensor): Converts message tensors to indices.
        _print_to_console(metrics): Prints the metrics to the console.
        get_results(): Returns the collected results.
    """
    
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
        """
        Called at the end of each epoch during training.

        Args:
            loss (float): The loss value at the end of the epoch.
            logs (core.Interaction): The interaction logs for the epoch.
            epoch (int): The current epoch number.
        """
        train_metrics = self._aggregate_metrics(loss, logs, "train", epoch)
        train_metrics["messages"] = self._messages_to_indices(logs.message)
        self.results.append(train_metrics)
        if self.print_train_loss:
            self._print_to_console({k: v for k, v in train_metrics.items() if k != 'messages'})

    def on_validation_end(self, loss: float, logs: core.Interaction, epoch: int):
        """
        Called at the end of each validation step.

        Args:
            loss (float): The loss value at the end of the validation step.
            logs (core.Interaction): The interaction logs for the validation step.
            epoch (int): The current epoch number.
        """
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
        """
        Aggregates the metrics for a given mode (train or test).

        Args:
            loss (float): The loss value.
            logs (core.Interaction): The interaction logs.
            mode (str): The mode (train or test).
            epoch (int): The current epoch number.

        Returns:
            dict: A dictionary containing the aggregated metrics.
        """
        metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        return {
            "epoch": epoch,
            "mode": mode,
            "loss": loss,
            **metrics
        }
    
    def _messages_to_indices(self, messages_tensor):
        """
        Converts message tensors to indices.

        Args:
            messages_tensor: The tensor containing the messages.

        Returns:
            list: A list of message indices.
        """
        return [message.argmax(dim=-1).tolist() for message in messages_tensor]

    def _print_to_console(self, metrics: dict):
        """
        Prints the metrics to the console.

        Args:
            metrics (dict): A dictionary containing the metrics to be printed.
        """
        output_message = ", ".join([f"{k}={v}" for k, v in metrics.items()])
        print(output_message, flush=True)

    def get_results(self):
        """
        Returns the collected results.

        Returns:
            list: A list of collected results.
        """
        return self.results