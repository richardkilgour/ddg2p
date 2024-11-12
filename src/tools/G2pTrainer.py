import threading
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.utils import test_on_subset


class G2pTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, optimizer, device, out_path, use_rpc=False,
                 test_subset=None):
        self.use_rpc = use_rpc
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.out_path = out_path
        self.test_subset = test_subset

    def _run_batch(self, x, t):
        # Zero your gradients for every batch!
        self.optimizer.zero_grad()
        # Make predictions for this batch
        outputs = self.model(x)
        # Compute the loss and its gradients
        loss = torch.nn.CrossEntropyLoss()(outputs.permute(0, 2, 1), t)
        loss.backward()
        # Adjust learning weights
        self.optimizer.step()

    def _find_rpc(self):
        rpc_list = ["zeta_0", "zeta_1"]
        for r in rpc_list:
            pass

    def _run_epoch(self, epoch):
        epoch_total_time = 0
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
                print(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\t{epoch=}\t{batch_num=}')
        else:
            for batch_num, (source, targets) in enumerate(self.dataloader):
                start_time = time.perf_counter()
                self._run_batch(source.to(self.device), targets.to(self.device))
                elapsed_time = time.perf_counter() - start_time
                epoch_total_time += elapsed_time
                if batch_num % 100 == 0:
                    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    print(f'{timestamp}\t{epoch=}\t{batch_num=}\t{source.shape=}\t{elapsed_time:.4f}')
        print(f'{epoch_total_time=:.4f}')

    def _test(self):
        start_time = time.perf_counter()
        print(f'Testing starts at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
        results = test_on_subset(self.test_subset, self.model, self.device, beam_width=1)
        testing_elapsed_time = time.perf_counter() - start_time
        print(f'Testing finished at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\t{testing_elapsed_time=:.4f}')
        return results

    def train(self, max_epochs):
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
            best_test_wer = 1.1  # 110% ensures the first epoch will improve and net will be saved
            best_test_per = 1.1
            no_improvement_count = 0
            for epoch in range(max_epochs):
                self._run_epoch(epoch)
                # After x epochs, evaluate on the test set
                if self.test_subset and epoch % 3 == 0:
                    test_ler, test_wer, test_per = self._test()
                    print(f'Tested {epoch=}\t{test_ler=}\t{test_wer=}\t{test_per=}')
                    if test_wer < best_test_wer or (test_wer == best_test_wer and test_per < best_test_per):
                        print(f'New best {test_wer=}')
                        no_improvement_count = 0
                        best_test_wer = test_wer
                        torch.save(self.model.state_dict(), self.out_path)
                    else:
                        no_improvement_count += 1
                        print(f'{test_wer=} is not better than {best_test_wer} - {no_improvement_count=}')
                    if no_improvement_count > 2:
                        print(f'OVERLEARNING??? after {epoch=}. ABORT training')
                        break
