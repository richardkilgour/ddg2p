import logging
import time
from datetime import datetime
from os.path import dirname

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# tensorboard --logdir=logs
# View Metrics: Open http://localhost:6006/

from src.data.DataUtils import test_on_subset, get_wer_per

logger = logging.getLogger(__name__)


class G2pTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, optimizer, device, out_path,
                 test_subset=None, train_reporting_cadence=100):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.out_path = out_path
        # Log directory for TensorBoard
        self.writer = SummaryWriter(dirname(out_path))
        self.test_subset = test_subset
        self.train_reporting_cadence = train_reporting_cadence

    def _run_batch(self, inputs, targets, batch_num, epoch):
        outputs = self.model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs.permute(0, 2, 1), targets)
        try:
            loss.backward()
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error(f'{epoch=} {batch_num=} FAILED with {e}\n{inputs.shape=}')
                # Optionally reduce the batch size or perform other cleanup
                # Just try again
                torch.cuda.empty_cache()
                loss.backward()
            else:
                raise e  # Re-raise if it's a different error
        return loss.item()

    def _run_epoch(self, epoch, update_after_epoch=False):
        self.model.train()
        epoch_total_time = 0
        epoch_loss = 0.
        self.optimizer.zero_grad()
        for batch_num, (source, targets) in enumerate(self.dataloader):
            start_time = time.perf_counter()
            batch_loss = self._run_batch(source.to(self.device), targets.to(self.device), batch_num, epoch)
            epoch_loss += batch_loss
            # Update after every batch
            if not update_after_epoch:
                self.optimizer.step()
                self.optimizer.zero_grad()
            elapsed_time = time.perf_counter() - start_time
            epoch_total_time += elapsed_time
            if batch_num % self.train_reporting_cadence == 0:
                timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                logger.info(
                    f'{timestamp}\t{epoch=}\t{batch_num=}\t{batch_loss=}\t{source.shape=}\t{elapsed_time:.4f}')
        if update_after_epoch:
            self.optimizer.step()
            self.optimizer.zero_grad()
        # Log batch loss
        self.writer.add_scalar('Loss/epoch train', epoch_loss)
        logger.info(f'{epoch_total_time=:.4f}')

        return epoch_loss / len(self.dataloader)

    def _test(self, epoch):
        """Run over the test set for the given epoch, and find a language confusion matrix and phoneme error rate"""
        start_time = time.perf_counter()
        logger.info(f'Testing starts at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        language_cm, test_per = test_on_subset(self.test_subset, self.model, beam_width=1)
        test_ler = 1. - language_cm.true_positive_rate()
        testing_elapsed_time = time.perf_counter() - start_time
        logger.info(f'Testing finished at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\t{testing_elapsed_time=:.4f}')

        # Log test metrics to tensorboard
        self.writer.add_scalar('Accuracy/test language error rate', test_ler, epoch)
        for language, per_rates in test_per.items():
            wer, per = get_wer_per(per_rates)
            self.writer.add_scalar(f'Accuracy/{language} test word error rate', wer, epoch)
            self.writer.add_scalar(f'Accuracy/{language} test phoneme error rate', per, epoch)

        return language_cm, test_per

    def train(self, max_epochs, training_metrics, early_stopping=10):
        epoch = training_metrics['epoch']
        best_test_wer = training_metrics['WER']
        best_test_per = training_metrics['PER']
        no_improvement_count = training_metrics['no_improvement_count']

        while epoch < max_epochs:
            loss = self._run_epoch(epoch)

            # Add epoch-level stats
            self.writer.add_scalar('Epoch Loss/train', loss, epoch)

            # After x epochs, evaluate on the test set
            if self.test_subset:
                language_cm, test_per = self._test(epoch)
                test_ler = 1. - language_cm.true_positive_rate()
                all_language_per = []
                for language, per_rates in test_per.items():
                    # One big list for all languages
                    all_language_per.extend(per_rates)
                    wer, per = get_wer_per(per_rates)
                    logger.info(f'{language=}\t{wer=:.5%}\t{per=:.5%}')

                test_wer, test_per = get_wer_per(all_language_per)

                logger.info(f'Overall\t{epoch=}\t{test_ler=:.5%}\t{test_wer=:.5%}\t{test_per=:.5%}')

                if test_wer < best_test_wer or (test_wer == best_test_wer and test_per < best_test_per):
                    logger.info(f'New best {test_wer=:.5%}')
                    no_improvement_count = 0
                    best_test_wer = test_wer
                    best_test_per = test_per
                    torch.save({
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'LER': test_ler,
                        'WER': test_wer,
                        'PER': test_per,
                        'train_loss': loss,
                        'no_improvement_count': no_improvement_count,
                    }, self.out_path + 'best_mamba_model.cp')
                else:
                    no_improvement_count += 1
                    logger.warning(f'{test_wer=:.5%}\t<{best_test_wer=:.5%}\t{no_improvement_count=}\t{best_test_per=:.5%}')
                    torch.save({
                        'epoch': epoch,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': self.optimizer.state_dict(),
                        'LER': test_ler,
                        'WER': test_wer,
                        'PER': test_per,
                        'train_loss': loss,
                        'no_improvement_count': no_improvement_count,
                    }, self.out_path + 'latest_mamba_model.cp')

                if no_improvement_count > early_stopping:
                    logger.warning(f'OVERLEARNING???\t{no_improvement_count=}\t{epoch=}\tABORT training\t{best_test_wer=:.5%}\t{best_test_per=:.5%}')
                    break
            epoch += 1
