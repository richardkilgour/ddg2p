import logging
import threading
import time
from datetime import datetime
from os.path import dirname

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# tensorboard --logdir=logs
# View Metrics: Open http://localhost:6006/

from src.data.DataUtils import test_on_subset

logger = logging.getLogger(__name__)


class G2pTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, optimizer, device, out_path, use_rpc=False,
                 test_subset=None, train_reporting_cadence=100):
        self.use_rpc = use_rpc
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
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs.permute(0, 2, 1), targets)
        loss.backward()
        self.optimizer.step()

        # Log batch loss
        self.writer.add_scalar('Loss/train', loss.item(), epoch * len(self.dataloader) + batch_num)
        return loss.item()

    def _find_rpc(self):
        rpc_list = ["zeta_0", "zeta_1"]
        for r in rpc_list:
            pass

    def _run_epoch(self, epoch):
        epoch_total_time = 0
        epoch_loss = 0
        if self.use_rpc:
            from torch.distributed.rpc import init_rpc, remote, rpc_async
            init_rpc("zeta_0", rank=0, world_size=2)
            rem = None
            for batch_num, (source, targets) in enumerate(self.dataloader):
                # NB: rpc_async does not support GPU tensors as arguments, so leave them on the CPU
                # fut = rpc_async("zeta_0", self._run_batch(source.to(self.device), targets.to(self.device)))
                if rem is None:
                    rem = remote("zeta_1", self._run_batch(source, targets))
                else:
                    fut = rpc_async("zeta_0", self._run_batch(source, targets))
                logger.info(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\t{epoch=}\t{batch_num=}')
        else:
            for batch_num, (source, targets) in enumerate(self.dataloader):
                start_time = time.perf_counter()
                batch_loss = self._run_batch(source.to(self.device), targets.to(self.device), batch_num, epoch)
                epoch_loss += batch_loss
                elapsed_time = time.perf_counter() - start_time
                epoch_total_time += elapsed_time
                if batch_num % self.train_reporting_cadence == 0:
                    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    logger.info(
                        f'{timestamp}\t{epoch=}\t{batch_num=}\t{batch_loss=}\t{source.shape=}\t{elapsed_time:.4f}')
        logger.info(f'{epoch_total_time=:.4f}')
        return epoch_loss / len(self.dataloader)

    def _test(self, epoch):
        start_time = time.perf_counter()
        logger.info(f'Testing starts at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        test_ler, test_wer, test_per = test_on_subset(self.test_subset, self.model, beam_width=1)
        testing_elapsed_time = time.perf_counter() - start_time
        logger.info(f'Testing finished at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\t{testing_elapsed_time=:.4f}')

        # Log test metrics
        self.writer.add_scalar('Accuracy/test language error rate', test_ler, epoch)
        self.writer.add_scalar('Accuracy/test word error rate', test_wer, epoch)
        self.writer.add_scalar('Accuracy/test phoneme error rate', test_per, epoch)

        return test_ler, test_wer, test_per

    def train(self, max_epochs, training_metrics, early_stopping=10):
        epoch = training_metrics['epoch']
        best_test_wer = training_metrics['WER']
        no_improvement_count = training_metrics['no_improvement_count']

        # If using RPC
        if self.use_rpc:
            from torch.distributed import rpc

            def cpu_client():
                rpc.init_rpc("zeta_1", rank=1, world_size=2)

            # Throw up a CPU thread for processing
            worker1 = threading.Thread(target=cpu_client)
            worker1.start()

            worker1.join()
        else:
            while epoch < max_epochs:
                loss = self._run_epoch(epoch)

                # Add epoch-level stats
                self.writer.add_scalar('Epoch Loss/train', loss, epoch)

                # After x epochs, evaluate on the test set
                if self.test_subset:
                    test_ler, test_wer, test_per = self._test(epoch)
                    logger.info(f'Tested {epoch=}\t{test_ler=:.5%}\t{test_wer=:.5%}\t{test_per=:.5%}')
                    if test_wer < best_test_wer or (test_wer == best_test_wer and test_per < best_test_per):
                        logger.info(f'New best {test_wer=}')
                        no_improvement_count = 0
                        best_test_wer = test_wer
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
                        logger.warning(f'{test_wer=} is not better than {best_test_wer} - {no_improvement_count=}')
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
                        logger.warning(f'OVERLEARNING??? after {epoch=}. ABORT training')
                        break
                epoch += 1
