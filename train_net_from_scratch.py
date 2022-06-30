""" train_net_from_scratch.py
    Train, test, and save neural networks without transfer learning
    Developed for Tabular Transfer Learning project
    March 2022
"""

import json
import logging
import os
import sys
from collections import OrderedDict

import hydra
import numpy as np
import torch
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

import deep_tabular as dt



# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115

@hydra.main(config_path="config", config_name="train_net_config")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_net_from_scratch.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.hyp.save_period < 0:
        cfg.hyp.save_period = 1e8
    torch.manual_seed(cfg.hyp.seed)
    torch.cuda.manual_seed_all(cfg.hyp.seed)
    writer = SummaryWriter(log_dir=f"tensorboard")

    ####################################################
    #               Dataset and Network and Optimizer
    loaders, unique_categories, n_numerical, n_classes = dt.utils.get_dataloaders(cfg)

    net, start_epoch, optimizer_state_dict = dt.utils.load_model_from_checkpoint(cfg.model,
                                                                                 n_numerical,
                                                                                 unique_categories,
                                                                                 n_classes,
                                                                                 device)
    pytorch_total_params = sum(p.numel() for p in net.parameters())

    log.info(f"This {cfg.model.name} has {pytorch_total_params / 1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")

    optimizer, warmup_scheduler, lr_scheduler = dt.utils.get_optimizer_for_single_net(cfg.hyp,
                                                                                      net,
                                                                                      optimizer_state_dict)
    criterion = dt.utils.get_criterion(cfg.dataset.task)
    train_setup = dt.TrainingSetup(criterions=criterion,
                                   optimizer=optimizer,
                                   scheduler=lr_scheduler,
                                   warmup=warmup_scheduler)
    ####################################################

    ####################################################
    #        Train
    log.info(f"==> Starting training for {max(cfg.hyp.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -np.inf
    done = False
    epoch = start_epoch
    best_epoch = epoch

    while not done and epoch < cfg.hyp.epochs:
        # forward and backward pass for one whole epoch handeld inside dt.default_training_loop()
        loss = dt.default_training_loop(net, loaders["train"], train_setup, device)
        log.info(f"Training loss at epoch {epoch}: {loss}")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        # TensorBoard writing
        writer.add_scalar("Loss/loss", loss, epoch)
        for i in range(len(optimizer.param_groups)):
            writer.add_scalar(f"Learning_rate/group{i}",
                              optimizer.param_groups[i]["lr"],
                              epoch)

        # evaluate the model periodically and at the final epoch
        if (epoch + 1) % cfg.hyp.val_period == 0 or epoch + 1 == cfg.hyp.epochs:
            test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                                   [loaders["test"], loaders["val"], loaders["train"]],
                                                                   cfg.dataset.task,
                                                                   device)
            log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
            log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
            log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")

            dt.utils.write_to_tb([train_stats["score"], val_stats["score"], test_stats["score"]],
                                 [f"train_acc-{cfg.dataset.name}",
                                  f"val_acc-{cfg.dataset.name}",
                                  f"test_acc-{cfg.dataset.name}"],
                                 epoch,
                                 writer)

        if cfg.hyp.use_patience:
            val_stats, test_stats = dt.evaluate_model(net,
                                                      [loaders["val"], loaders["test"]],
                                                      cfg.dataset.task,
                                                      device)
            if val_stats["score"] > highest_val_acc_so_far:
                best_epoch = epoch
                highest_val_acc_so_far = val_stats["score"]
                log.info(f"New best epoch, val score: {val_stats['score']}")
                # save current model
                state = {"net": net.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()}
                out_str = "model_best.pth"
                log.info(f"Saving model to: {out_str}")
                torch.save(state, out_str)

            if epoch - best_epoch > cfg.hyp.patience:
                done = True
        epoch += 1
        writer.flush()
        writer.close()

    log.info("Running Final Evaluation...")
    checkpoint_path = "model_best.pth"
    net.load_state_dict(torch.load(checkpoint_path)["net"])
    test_stats, val_stats, train_stats = dt.evaluate_model(net,
                                                           [loaders["test"], loaders["val"], loaders["train"]],
                                                           cfg.dataset.task,
                                                           device)

    log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
    log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
    log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")

    stats = OrderedDict([("dataset", cfg.dataset.name),
                         ("model_name", cfg.model.name),
                         ("run_id", cfg.run_id),
                         ("best_epoch", best_epoch),
                         ("routine", "from_scratch"),
                         ("test_stats", test_stats),
                         ("train_stats", train_stats),
                         ("val_stats", val_stats)])
    with open(os.path.join("stats.json"), "w") as fp:
        json.dump(stats, fp, indent=4)
    log.info(json.dumps(stats, indent=4))
    ####################################################
    return stats


if __name__ == "__main__":
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()
