# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings

import mmcv
from mmcv.runner import EpochBasedRunner, IterBasedRunner
from mmcv.runner.utils import get_host_info
from mmcv.runner import IterLoader
from mmcv.runner.builder import RUNNERS
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np

@RUNNERS.register_module()
class InfiniteEpochBasedRunner(EpochBasedRunner):
    """Epoch-based Runner supports dataloader with InfiniteSampler.

    The workers of dataloader will re-initialize, when the iterator of
    dataloader is created. InfiniteSampler is designed to avoid these time
    consuming operations, since the iterator with InfiniteSampler will never
    reach the end.
    """
    # TODO: add tensor_writer
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super().__init__(model, batch_processor, optimizer, work_dir, logger, meta, max_iters, max_epochs)
        if work_dir is not None and work_dir == '/export':
            self.tensor_writer = SummaryWriter('./tensorboard_log/')
        else:
            self.tensor_writer = SummaryWriter(f'{work_dir}/tensorboard_log')

    # TODO: derived from EpochBasedRunner
    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
    
    # TODO: derived from EpochBasedRunner
    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         mmcv.runner.get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            self.tensor_writer.add_scalar('progress', self.epoch / self._max_epochs, global_step=self.epoch)

            mode, epochs = flow
            if isinstance(mode, str):  # self.train() below
                if not hasattr(self, mode):
                    raise ValueError(
                        f'runner has no method named "{mode}" to run an '
                        'epoch')
                epoch_runner = getattr(self, mode)
            else:
                raise TypeError(
                    'mode in workflow must be a str, but got {}'.format(
                        type(mode)))
            for _ in range(epochs):
                if mode == 'train' and self.epoch >= self._max_epochs:
                    break
                epoch_runner(data_loaders[i], **kwargs)
            if self.epoch % 2 == 0:
                log_vars = self.outputs['log_vars']
                for log_var in log_vars:
                    self.tensor_writer.add_scalar(log_var, log_vars[log_var], global_step=self.epoch)
                self.tensor_writer.flush()

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def train(self, data_loader: DataLoader, **kwargs) -> None:
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition

        # To reuse the iterator, we only create iterator once and bind it
        # with runner. In the next epoch, the iterator will be used against
        if not hasattr(self, 'data_loader_iter'):
            self.data_loader_iter = iter(self.data_loader)

        # The InfiniteSampler will never reach the end, but we set the
        # length of InfiniteSampler to the actual length of dataset.
        # The length of dataloader is determined by the length of sampler,
        # when the sampler is not None. Therefore, we can simply forward the
        # whole dataset in a epoch by length of dataloader.

        for i in range(len(self.data_loader)):
            data_batch = next(self.data_loader_iter)
            self._inner_iter = i
            self.call_hook('before_train_iter')
            self.run_iter(data_batch, train_mode=True, **kwargs)
            self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1

@RUNNERS.register_module()
class IterBasedRunnerWithLog(IterBasedRunner):
    
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super().__init__(model, batch_processor, optimizer, work_dir, logger, meta, max_iters, max_epochs)
        if work_dir is not None and work_dir == '/export':
            self.tensor_writer = SummaryWriter('./tensorboard_log/')
        else:
            self.tensor_writer = SummaryWriter(f'{work_dir}/tensorboard_log')
    
    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            self.tensor_writer.add_scalar('progress', self.iter / self._max_iters, global_step=self.iter)
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)
            if self.iter % 200 == 0:
                log_vars = self.outputs['log_vars']
                for log_var in log_vars:
                    self.tensor_writer.add_scalar(log_var, log_vars[log_var], global_step=self.iter)
                    self.tensor_writer.flush()

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')