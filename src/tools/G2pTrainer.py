import threading
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.utils import test_on_subset


class G2pTrainer:
    def __init__(self, model: nn.Module, dataloader: DataLoader, optimizer, device, out_path, use_rpc=False, test_subset=None):
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
                print(f'{epoch=}; {batch_num=}')
        else:
            for batch_num, (source, targets) in enumerate(self.dataloader):
                start_time = time.perf_counter()
                self._run_batch(source.to(self.device), targets.to(self.device))
                elapsed_time = time.perf_counter() - start_time
                epoch_total_time += elapsed_time
                if batch_num % 20 == 0:
                    print(f'{epoch=}\t{batch_num=}\t{(source.shape)=}\t{elapsed_time:.4f}')
                if batch_num > 50000:
                    break
        print(f'{epoch_total_time=:.4f}')
        # After an Epoch, evaluate on the test set
        start_time = time.perf_counter()
        print('Testing...')
        results = test_on_subset(self.test_subset, self.model, self.device)
        testing_elapsed_time = time.perf_counter() - start_time
        print(f'{testing_elapsed_time=:.4f}')
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
            best_test_WER = 1.1 # 110% ensures the first epoch will improve and net will be saved
            best_test_PER = 1.1
            no_improvement_count = 0
            for epoch in range(max_epochs):
                correct_language, correct_phoneme, total_PER = self._run_epoch(epoch)
                WER = 1. - correct_phoneme
                print(f'Tested {epoch=}\t{correct_language=}\t{WER=}\t{total_PER=}')
                if WER < best_test_WER or (WER == best_test_WER and total_PER < best_test_PER):
                    print(f'New best test {WER=}')
                    no_improvement_count = 0
                    torch.save(self.model.state_dict(), self.out_path)
                else:
                    no_improvement_count +=1
                    print(f'{WER=} is not better than {best_test_WER} - {no_improvement_count=}')
                if no_improvement_count > 2:
                    print(f'OVERLEARNING??? after {epoch=}')
                    break

