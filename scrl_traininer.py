import gc
import logging

import torch
import torch.nn.functional as F

from models.scrl import SCRLBoxGenerator
from models.scrl import SingleLayerLinearHead, TwoLayerLinearHead

torch.backends.cudnn.benchmark = True
C = Colorer.instance()


def _release_memory(*objects):
    del objects
    gc.collect()
    torch.cuda.empty_cache()


class SCRLTrainer():
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if self.cfg.network.scrl.enabled:
            self.box_generator = SCRLBoxGenerator.init_from_config(cfg)
        
    def run(self):
        log.info(C.green("[!] Start the Trainer."))
        result_pack = ResultPack(exp_name=self.cfg.config_name)
        
        if self.target_network is not None:
            self._initialize_target_network(from_online=False)
            
        # resume from the chekcpoint if possible
        self.load_checkpoint_if_available()
        if self.cfg.train.enabled and self.cur_epoch == self.max_epochs:
            self.cfg.defrost()
            self.cfg.train.enabled = False
            self.cfg.freeze()

        if not self.cfg.train.enabled:
            self.max_epochs = 0
            log.info(C.green('[!] Load Pre-trained Parameters.'))
        else:
            method = 'SCRL' if self.cfg.network.scrl.enabled else 'BYOL'
            log.info(C.green(f"[!] Upstream: {method} Pre-training."))
            eta = TimeOfArrivalEstimator.init_from_epoch_steps(
                epochs=self.max_epochs - self.cur_epoch,
                epoch_steps=len(self.train_loader),
            )
                
        comm.synchronize()
        # Upstream: BYOL or SCRL
        for epoch in range(1 + self.cur_epoch, self.max_epochs + 1):
            self.cur_epoch = epoch
            self.train_loader.sampler_origin.set_epoch(epoch)
            disp = DistributedProgressDisplayer(
                max_steps=len(self.train_loader), 
                no_pbar=self.cfg.no_pbar,
                desc=(f"{C.selected('[Upstream:Train]')} "
                      f"{C.underline(f'[{self.cfg.save_dir}]')}")
            )
                
            for step, (views, labels) in enumerate(self.train_loader, start=1):
                
                # gear up inputs (and spatially consistent boxes if needed)
                if self.cfg.network.scrl.enabled:
                    views, transf, _, _ = decompose_collated_batch(views)
                    boxes = self.box_generator.generate(transf)
                else:
                    boxes =  None
                    
                # model forward and loss computation
                with detect_anomaly_with_redirected_cpp_error(self.cfg.detect_anomaly):
                    with torch.cuda.amp.autocast(not self.cfg.disable_autocast):
                        outs = self._forward(views, labels, boxes)
                        loss_total = outs.by_cls_name('Loss').weighted_sum_scalars()
                        
                # optimization
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.scaler.scale(loss_total).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # log
                disp.update_with_postfix(
                    f"ep:{epoch}/{self.max_epochs}, "
                    f"lr:{self.scheduler.get_last_lr()[0]:5.4f}, "
                    f"m:{self.m:5.4f}, "
                    f"{str(outs.by_cls_name('Loss'))}, "
                    f"{str(outs.by_cls_name('Metric'))}, "
                    f"eEvaMax:{self.max_eval_score:5.2f}%, "
                    f"eta:{eta.estimate_step_str()}"
                )
                global_step = len(self.train_loader) * (epoch - 1) + step - 1
                self.tb_writer.add_outputs(outs, global_step)
                
                # EMA update
                comm.synchronize()
                if self.target_network is not None:
                    self._decay_ema_momentum(global_step)
                    self._update_target_network_parameters()

            # end of each epoch
            disp.close()
                
            # save model on a regular basis
            if (comm.is_main_process() and 
                    (epoch % self.cfg.train.snapshot_interval == 0 or 
                     epoch == self.max_epochs)):
                self.save_checkpoint(epoch)
                self.symlink_checkpoint_with_tag(epoch, 'last')

            # online evaluation
            if self.cfg.train.online_eval:
                last_eval_on = linear_eval_online(
                    cfg=self.cfg,
                    epoch=self.cur_epoch,
                    eval_loader=self.eval_loader,
                    backbone=self.online_network,
                    head=self.evaluator)
                
                is_best = ""
                if (last_eval_on and last_eval_on > self.max_eval_score):
                    self.max_eval_score = last_eval_on
                    self.max_eval_epoch = epoch 
                    if comm.is_main_process():
                        self.save_best_checkpoint()
                    # self.symlink_best_checkpoint(epoch)
                    is_best = C.red("[<- Best Acc.]")
            
                log.info(
                    f"{C.red('[Eval result]')} "
                    f"ep:{epoch}/{self.max_epochs}, "
                    f"eEvalAcc:{last_eval_on:5.2f}%, "
                    f"eEvaMax:{self.max_eval_score:5.2f}% {is_best}"
                )

            if comm.synchronize() and comm.is_local_main_process():
                result_dict = outs.scalar_only().to_dict()
                result_dict.update({'eEvaMax': self.max_eval_score})
                result_pack.append(epoch, **result_dict)
                yield TaskReturns(state=TaskState.TRAIN, 
                                  value={'pack': result_pack})

        # end of upstream
        if self.cfg.train.enabled and comm.is_local_main_process():
            final_result = (
                f"[Final result (pre-training)] [{self.cfg.save_dir}] "
                f"{str(outs.by_cls_name('Loss'))}, "
                f"eEvalAcc:{last_eval_on:5.3f}%, "
                f"eEvaMax:{self.max_eval_score:5.3f}% @{self.max_eval_epoch}ep."
            )
            log.info(C.cyan(final_result))
            log_result.info(final_result)
            
        # Downstream: linear evaluation
        if self.cfg.eval.enabled:
            if comm.is_local_main_process():
                # yield the state here so as to update progress bar, etc., since
                # no returns there will be until evaluation is fully completed.
                # (Note that this comment is relvant to our internal API)
                yield TaskReturns(state=TaskState.EVAL)

            log.info(C.green(f"[!] Downstream: Linear Evaluation."))
            last_eval_off, max_eval_off, max_eval_epoch = linear_eval_offline(
                cfg=self.cfg,
                backbone=self.online_network,
                finetune=self.cfg.eval.finetune)
            
            if comm.synchronize() and comm.is_local_main_process():
                final_result = (
                    f"[Final result (linear eval.)] [{self.cfg.save_dir}] "
                    f"last:{last_eval_off:5.3f}%, "
                    f"max:{max_eval_off:5.3f}% @{max_eval_epoch}ep."
                )
                log.info(C.cyan(final_result))
                log_result.info(final_result)
            
            log.info("[Save] the final results are saved in: ./log_result.txt.")

        if comm.is_local_main_process():
            yield TaskReturns(state=TaskState.DONE) 

        if self.cfg.spawn_ctx:
            self.cfg.spawn_ctx.join()

        log.info(C.green("[!] End of the Trainer."))
        

    def _forward(self, views, labels, boxes=None):
        # compose mini-batch of views
        views = torch.cat(views, dim=0)
        views = views.to(self.device, non_blocking=True)

        # compose mini-batch of the matched box coords.
        if boxes is not None:
            boxes = torch.cat(boxes, dim=0)
            boxes = boxes.to(self.device, non_blocking=True)

        # online network 
        p_online, h_online = self.online_network(views, boxes)

        # return None by default
        byol_loss, scrl_loss, eval_loss, eval_acc = (None,) * 4
        
        # target network
        with torch.no_grad():
            p_target, _ = self.target_network(views, boxes)

        if self.cfg.network.scrl.enabled:
            # SCRL loss
            p_online = self.predictor(p_online)
            scrl_loss = self._criterion(p_online, p_target)
        else:
            # BYOL loss
            p_online = self.predictor(p_online)
            byol_loss = self._criterion(p_online, p_target)

        # online evaluator loss
        if self.cfg.train.online_eval:
            labels = torch.cat((labels,) * 2, dim=0)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.evaluator(h_online.detach())  # stop gradient
            eval_loss = self.xent_loss(logits, labels)
            preds = torch.argmax(logits, dim=1)
            eval_acc = (preds == labels).float().mean().item()
            
        return TrainerOutputs(*[
            Loss(name='bLoss', value=byol_loss, fmt="4.3f"),
            Loss(name='sLoss', value=scrl_loss, fmt="4.3f"),
            Loss(name='eLoss', value=eval_loss, fmt="4.3f"),
            Metric(name='eAcc', value=eval_acc, fmt="5.2f", 
                   weight=100., suffix="%"),
        ])

def _unwrap(wrapped_module):
    if isinstance(wrapped_module, DistributedDataParallel):
        module = wrapped_module.module
    else:
        module = wrapped_module
    return module


def _regression_loss(x, y):
        # eps = 1e-6 if torch.is_autocast_enabled() else 1e-12
        x = F.normalize(x, p=2, dim=1) #, eps=eps)
        y = F.normalize(y, p=2, dim=1) #, eps=eps)
        return (2 - 2 * (x * y).sum(dim=1)).view(-1)


class BYOLBasedTrainer:
    """This trainer supports BYOL-like training framework that can be subclassed 
    by other task-specific trainer classes. To specify a detailed algorithm, 
    the user should implement Traniner.run().
    """
    def __init__(self, cfg, online_network, target_network, 
                 predictor=None, evaluator=None,
                 train_loader=None, eval_loader=None):
        if cfg.train.enabled:
            assert train_loader is not None
            assert predictor is not None
        if cfg.train.enabled and cfg.train.online_eval:
            assert eval_loader is not None
            assert evaluator is not None
            
        self._modules = {}
        self._saving_targets = {}
        self.cfg = cfg
        self.device = cfg.device
        
        self.online_network = online_network
        self.target_network = target_network
        self.predictor = predictor
        self.evaluator = evaluator
        self.xent_loss = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self._setup_device_and_distributed_parallel(cfg.device)

        self.cur_epoch = 0
        self.max_epochs = 0
        self.max_eval_score = 0.
        self.max_eval_epoch = 0
        
        if self.cfg.train.enabled:
            self.m_base = self.m = cfg.train.m
            self.max_epochs = cfg.train.max_epochs
            self.total_global_step = len(train_loader) * cfg.train.max_epochs
            self.optimizer, self.scheduler = get_optimizer_and_scheduler(
                cfg=self.cfg, mode='train', modules=self._modules, loader=train_loader,
                exclude_from_lars=True, module_black_list=['target_network'])
            self.scaler = torch.cuda.amp.GradScaler() #init_scale=2**14)
            # default init_scale 2**16 will yield invalid gradient in the first interation 
            self.tb_writer = TensorBoardWriter.init_for_train_from_config(cfg)
        else:
            self.optimizer, self.scheduler, self.scaler = None, None, None

    def __setattr__(self, name, value):
        if hasattr(value, 'state_dict') and callable(value.state_dict):
            self._saving_targets[name] = value  # including optimzers & schedulers
        if isinstance(value, nn.Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)
        
    def run(self):
        """Main training algorithm should be implemented in this method."""
        raise NotImplementedError()
    
    @classmethod
    def init_from_config(cls, cfg):
        train_loader, eval_loader, num_classes = get_loaders_for_trainer(cfg)
        online_network = Backbone.init_from_config(cfg)
        target_network, predictor, evaluator = None, None, None
        if cfg.train.enabled:
            target_network = Backbone.init_from_config(cfg)
            predictor = TwoLayerLinearHead.init_predictor_from_config(cfg)
            evaluator = SingleLayerLinearHead.init_evaluator_from_config(
                cfg, num_classes)
        return cls(
            cfg=cfg,
            train_loader=train_loader,
            eval_loader=eval_loader,
            online_network=online_network,
            target_network=target_network,
            predictor=predictor,
            evaluator=evaluator,
        )

    def _setup_device_and_distributed_parallel(self, device):
        for name, module in self._modules.items():
            module = module.to(device)
            module = utils.wrap_if_distributed(module, device)
            self._modules[name] = module
            object.__setattr__(self, name, module)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), 
                                    self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def _decay_ema_momentum(self, step):
        self.m = (1 - (1 - self.m_base) * 
                  (math.cos(math.pi * step / self.total_global_step) + 1) / 2)

    @staticmethod
    def _criterion(p_online, p_target):
        """Regression loss used in BYOL."""
        p_online_v1, p_online_v2 = p_online.chunk(2)
        p_target_v1, p_target_v2 = p_target.chunk(2)
        assert p_online_v1.size(0) == p_online_v2.size(0)
        assert p_target_v1.size(0) == p_target_v2.size(0)
        assert p_online_v1.size(0) == p_target_v1.size(0)
        # symmetric loss
        loss = _regression_loss(p_online_v1, p_target_v2)
        loss += _regression_loss(p_online_v2, p_target_v1)
        return loss.mean()

    def _initialize_target_network(self, from_online):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), 
                                    self.target_network.parameters()):
            if from_online:
                param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def _save_checkpoint(self, tag):
        save_path = f"{self.cfg.save_dir}/checkpoint_" + str(tag) + ".pth"
        state_dict = {
            'tag': str(tag), 
            'epoch': self.cur_epoch,
            'max_eval_score': self.max_eval_score,
            'max_eval_epoch': self.max_eval_epoch,
            }
        for key, target in self._saving_targets.items():
            if self.cfg.fake_checkpoint:
                target = "fake_state_dict"
            else:
                target = utils.unwrap_if_distributed(target)
                target = target.state_dict()
            state_dict[f"{key}_state_dict"] = target

        torch.save(state_dict, save_path)
        suffix = (C.debug(" (fake_checkpoint)") 
                  if self.cfg.fake_checkpoint else "")
        return save_path + suffix

    def save_checkpoint(self, epoch):
        save_path = self._save_checkpoint(str(epoch))
        log.info(f"[Save] restore the model's checkpoint: {save_path}")
        return save_path
    
    def save_best_checkpoint(self):
        save_path = self._save_checkpoint('best')
        log.info(f"[Save] restore the best model's checkpoint: {save_path}")
        return save_path

    def symlink_checkpoint_with_tag(self, epoch, tag):
        save_path = f"{self.cfg.save_dir}/checkpoint_{epoch}.pth"
        symlink_path = f"{self.cfg.save_dir}/checkpoint_{tag}.pth"
        if not os.path.exists(save_path):
            self._save_checkpoint(epoch)
        try:
            os.symlink(os.path.abspath(save_path), symlink_path)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(symlink_path)
                os.symlink(os.path.abspath(save_path), symlink_path)
            else:
                raise e
        finally:
            log.info(f"[Save] make a symlink of the current model: "
                     f"{symlink_path}")
        return symlink_path

    def load_checkpoint_if_available(self, tag='last'):
        if self.cfg.overwrite:
            assert not self.cfg.load_dir, \
                "Mutually exclusive aruguements: overwrite, load_dir."
            log.warning("Overwrite checkpoints in save_dir.")
            return False
        try:
            load_dir = self.cfg.load_dir or self.cfg.save_dir
            load_path = f"{load_dir}/checkpoint_{tag}.pth"
            state_dict = torch.load(load_path)
        except FileNotFoundError:
            if self.cfg.load_dir:
                raise FileNotFoundError(f"Can't find checkpoint at {load_dir}")
            else:
                log.warning(f'No checkpoint to resume from {load_dir}.')
            return False

        self.cur_epoch = state_dict['epoch']
        self.max_eval_score = state_dict['max_eval_score']
        self.max_eval_epoch = state_dict['max_eval_epoch']
        state_dict = {k[:-len('_state_dict')]: v for k, v in state_dict.items() 
                      if k.endswith('_state_dict')}
        log.info(f"[Resume] Loaded chekpoint (epoch: {self.cur_epoch}) "
                 f"from: {load_path}")

        missing_keys = set(self._saving_targets.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(self._saving_targets.keys())
        assert len(missing_keys) == 0, "Missing keys!"
        log.info("[Resume] Redundant keys: "
                 f"{list(unexpected_keys) if unexpected_keys else 'None'}")

        for key, target in self._saving_targets.items():
            if state_dict[key] == 'fake_state_dict':
                log.info(f"[Resume] Loaded {key}: {C.debug('(fake_chekpoint)')}")
            else:
                kwargs = {'strict': False} if isinstance(target, nn.Module) else {}
                loaded = _unwrap(target).load_state_dict(state_dict[key], **kwargs)
                if isinstance(target, nn.Module):
                    assert len(loaded.missing_keys) == 0
                if isinstance(target, Backbone):
                    # the projector is be ignored in evaluation-only cases
                    assert all([key.startswith('projector.') 
                                for key in loaded.unexpected_keys])
                log.info(f"[Resume] Loaded {key}")
        return True

def _get_desc(mode, save_dir):
    head = C.selected(f'[{mode}]')
    save_dir = C.underline(f'[{save_dir}]')
    return f"{head} {save_dir}"


@ExceptionLogger("error")
@torch.no_grad()  # the order \btw decorators matters
def iter_eval_epoch(
    cfg: Config, backbone: Backbone, head: SingleLayerLinearHead, 
    loader: DataLoader, criterion=None, finetune=False):
    """Work on evaluation mode when criterion=None."""
    
    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        labels = y.to(cfg.device, non_blocking=True)
        
        # feedforward
        with autocast(not cfg.disable_autocast):
            
            with enable_grad() if criterion and finetune else ExitStack():
                h = backbone(x, boxes=None, no_projection=True)
                
            with enable_grad() if criterion else ExitStack():
                logits = head(h if finetune else h.detach())
                loss = criterion(logits, labels) if criterion else 0.
                
        yield logits, labels, loss
        

@ExceptionLogger("error")
def linear_eval_online(cfg: Config, epoch: int, eval_loader: DataLoader, 
                       backbone: Backbone, head: SingleLayerLinearHead):
    assert eval_loader is not None
    assert head is not None
    backbone.eval()
    n_correct = 0
    n_samples = 0
    
    with DistributedProgressDisplayer(
            max_steps=len(eval_loader), 
            no_pbar=cfg.no_pbar,
            desc=_get_desc('Upstream: Eval', cfg.save_dir)
        ) as disp:
    
        for logits, labels, _ in iter_eval_epoch(
                cfg=cfg,backbone=backbone, head=head, 
                loader=eval_loader, criterion=None
            ):
            preds = torch.argmax(logits, dim=1)
            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()
            acc = (preds == labels).float().mean().item() * 100
            
            disp.update_with_postfix(
                f"ep:{epoch}/{cfg.train.max_epochs}, "
                f"#samples:{n_samples:5d}, "
                f"Acc: {acc:5.4f}"
            )

    backbone.train()
    acc = n_correct / n_samples * 100
    acc = sync_weighted_mean(acc, n_samples)

    return acc
        

@ExceptionLogger("error")
def linear_eval_offline(cfg: Config, backbone: nn.Module, finetune=False):
    train_loader, eval_loader, num_classes = get_loaders_for_linear_eval(cfg)
    head = SingleLayerLinearHead.init_evaluator_from_config(cfg, num_classes)
    head = head.to(unwrap_if_distributed(backbone).device)
    if comm.get_local_size() > 1:
        head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(head)
        head = DistributedDataParallel(module=head, 
                                       device_ids=[cfg.device], 
                                       broadcast_buffers=False, 
                                       find_unused_parameters=True)
    modules = {'head': head}
    if finetune:
        log.info("Fine-tuning mode.")
        backbone.train()
        modules.update({'backbone': backbone})
    else:
        # to use running statistics for the frozen backbone
        backbone.eval()
        
    optimizer, scheduler = get_optimizer_and_scheduler(
        cfg=cfg, mode='eval', modules=modules, loader=train_loader)
    scaler = torch.cuda.amp.GradScaler()

    max_eval_acc = 0.
    max_eval_epoch = 0
    max_epochs = cfg.eval.max_epochs
    criterion = torch.nn.CrossEntropyLoss()
    eta = TimeOfArrivalEstimator.init_from_epoch_steps(
        epochs=cfg.eval.max_epochs,
        epoch_steps=len(train_loader),
    )
    
    # training & validation loop
    for epoch in range(1, max_epochs + 1):
        train_loader.sampler_origin.set_epoch(epoch)
            
        with DistributedProgressDisplayer(
                max_steps=len(train_loader), 
                no_pbar=cfg.no_pbar,
                desc=_get_desc('Downstream:Train', cfg.save_dir)
            ) as disp:
            
            for logits, labels, loss in iter_eval_epoch(
                    cfg=cfg, backbone=backbone, head=head, loader=train_loader, 
                    criterion=criterion, finetune=finetune
                ):
                scheduler.step()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                
                preds = torch.argmax(logits, dim=1)
                acc = (preds == labels).float().mean().item() * 100
                comm.synchronize()

                disp.update_with_postfix(
                    f"ep:{epoch}/{max_epochs}, "
                    f"lr:{scheduler.get_last_lr()[0]:5.4f}, "
                    f"Loss:{loss:5.4f}, "
                    f"Acc:{acc:5.2f}%, "
                    f"EvalMax:{max_eval_acc:5.2f}%, "
                    f"eta:{eta.estimate_step_str()}"
                )

        # end of each epoch
        del logits, labels, loss

        is_valid_step = (cfg.eval.valid_interval 
                         and epoch % cfg.eval.valid_interval == 0)
        if not (is_valid_step or epoch == max_epochs) or epoch < 1:
            continue

        # validation
        n_correct = 0
        n_samples = 0
        backbone.eval()
        
        with DistributedProgressDisplayer(
                max_steps=len(eval_loader), 
                no_pbar=cfg.no_pbar,
                desc=_get_desc('Downstream: Eval', cfg.save_dir),
            ) as disp:
        
            for logits, labels, _ in iter_eval_epoch(
                    cfg=cfg, backbone=backbone, head=head, 
                    loader=eval_loader, criterion=None
                ):
                
                preds = torch.argmax(logits, dim=1)
                n_samples += labels.size(0)
                n_correct += (preds == labels).sum().item()
                acc = (preds == labels).float().mean().item() * 100
                
                disp.update_with_postfix(
                    f"ep:{epoch}/{max_epochs}, "
                    f"#samples:{n_samples:5d}, "
                    f"Acc:{acc:5.2f}%, "
                    f"EvalMax:{max_eval_acc:5.2f}%"
                )
                    
        last_eval_acc = n_correct / n_samples * 100
        last_eval_acc = sync_weighted_mean(last_eval_acc, n_samples)

        is_best = ""
        if last_eval_acc > max_eval_acc:
            max_eval_acc = last_eval_acc
            max_eval_epoch = epoch
            is_best = C.red("[<- Best Acc.]")

        if comm.synchronize() and comm.is_local_main_process():
            log.info(
                f"{C.red('[Eval result]')} "
                f"ep:{epoch}/{max_epochs}, "
                f"EvalAcc:{last_eval_acc:5.2f}%, "
                f"EvalMax:{max_eval_acc:5.2f}% {is_best}"
            )
        
        if finetune:
            backbone.train()

    return last_eval_acc, max_eval_acc, max_eval_epoch