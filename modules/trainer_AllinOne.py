import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
from tqdm import tqdm
import torch.distributed as dist
import logging


class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        self.args = args

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # setup GPU device if available, move model into configured device
        # self.device, device_ids = self._prepare_device(args.n_gpu)
        # self.model = model.to(self.device)
        # if len(device_ids) > 1:
        #     self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.model = model
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        # if not os.path.exists(self.checkpoint_dir):
        #     os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch, model_name):
        raise NotImplementedError

    def train(self, rank):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            print("epoch: ", epoch)
            self.train_dataloader.sampler.set_epoch(epoch)
            result = self._train_epoch(epoch, self.args.model_name)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('{} \t{:15s}: {}'.format(rank,str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning(
                        "Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    print(f'mnt_best: {self.mnt_best}, mnt_metric: {log[self.mnt_metric]}')
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1
                    best = False

                print(f'improved: {improved}, not_improved_count: {not_improved_count} mnt_best: {self.mnt_best}')

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0 and rank==0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        #* Code update since pandas is new version
        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['val']])], ignore_index=True)
        record_table = pd.concat([record_table, pd.DataFrame([self.best_recorder['test']])], ignore_index=True)
        record_table.to_csv(record_path, index=False)
    
    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")
    
    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        if self.mnt_metric_test in log:
            improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
                self.mnt_metric_test]) or \
                            (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                                self.mnt_metric_test])
            if improved_test:
                self.best_recorder['test'].update(log)
    
    def _print_best(self):
        self.logger.info('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))

        self.logger.info('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            self.logger.info('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader)
        # self.lr_scheduler = lr_scheduler
        # self.train_dataloader = train_dataloader
        # self.val_dataloader = val_dataloader
        # self.test_dataloader = test_dataloader
        self.args = args
        # self.device = 'cuda'

    def _train_epoch(self, epoch, model_name):

        self.logger.info('[{}/{}] Start to train in the training set.'.format(epoch, self.epochs))
        dist.barrier()
        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.train_dataloader, desc='Training')):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = self.model(images, reports_ids, mode = 'train')
            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            
            if batch_idx % self.args.log_period == 0:
                self.logger.info('[{}/{}] Step: {}/{}, Training Loss: {:.5f}.'
                                 .format(epoch, self.epochs, batch_idx, len(self.train_dataloader),
                                         train_loss / (batch_idx + 1)))
            
        log = {'train_loss': train_loss / len(self.train_dataloader)}
        
        self.logger.info('[{}/{}] Start to evaluate in the validation set.'.format(epoch, self.epochs))
        dist.barrier()
        self.model.eval()
        with torch.no_grad():
            val_gts_ids, val_res_ids = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.val_dataloader, desc='Validating')):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')

                val_res_ids.append(output)  # predict
                val_gts_ids.append(reports_ids)  # ground truth

            val_res_ids = distributed_concat(torch.cat(val_res_ids, dim=0),
                                             len(self.val_dataloader.dataset)).cpu().numpy()
            val_gts_ids = distributed_concat(torch.cat(val_gts_ids, dim=0),
                                             len(self.val_dataloader.dataset)).cpu().numpy()

            val_gts, val_res = self.model.module.tokenizer.decode_batch(val_gts_ids[:,1:]), self.model.module.tokenizer.decode_batch(val_res_ids)

            val_res = list(map(lambda x: 'placeholder' if x.strip() == '' else x, val_res))
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.logger.info('[{}/{}] Start to evaluate in the test set.'.format(epoch, self.epochs))
        if epoch % 10==0:
            dist.barrier()
            self.model.eval()
            with torch.no_grad():
                test_gts_ids, test_res_ids = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(tqdm(self.test_dataloader, desc='Testing')):
                    images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                    output = self.model(images, mode='sample')

                    test_res_ids.append(output)  # predict
                    test_gts_ids.append(reports_ids)  # ground truth

                test_res_ids = distributed_concat(torch.cat(test_res_ids, dim=0),
                                                  len(self.test_dataloader.dataset)).cpu().numpy()
                test_gts_ids = distributed_concat(torch.cat(test_gts_ids, dim=0),
                                                  len(self.test_dataloader.dataset)).cpu().numpy()

                test_gts, test_res = self.model.module.tokenizer.decode_batch(test_gts_ids[:,1:]), self.model.module.tokenizer.decode_batch(test_res_ids)

                test_res = list(map(lambda x: 'placeholder' if x.strip() == '' else x, test_res))
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log

def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]