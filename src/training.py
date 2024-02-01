import logging
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
from prettytable import PrettyTable
from torch.nn.functional import binary_cross_entropy_with_logits, multilabel_margin_loss

from util import ddi_rate_score, multi_label_metric

logger = logging.getLogger(__name__)


def eval_one_epoch(model, data_eval, voc_size, drug_data):
    model = model.eval()
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0
    for step, input_seq in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        for adm_idx, adm in enumerate(input_seq):
            output, _ = model(patient_data=input_seq[: adm_idx + 1], **drug_data)
            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            output = torch.sigmoid(output).detach().cpu().numpy()[0]
            y_pred_prob.append(output)

            y_pred_tmp = output.copy()
            y_pred_tmp[y_pred_tmp >= 0.5] = 1
            y_pred_tmp[y_pred_tmp < 0.5] = 0
            y_pred.append(y_pred_tmp)

            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(sorted(y_pred_label_tmp))
            visit_cnt += 1
            med_cnt += len(y_pred_label_tmp)
        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(
            np.array(y_gt), np.array(y_pred), np.array(y_pred_prob)
        )
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)

    ddi_rate = ddi_rate_score(smm_record, path="data/ddi_A_final.pkl")
    return (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )


def Test(model, model_path, device, data_test, voc_size, drug_data):
    test_metric_header = ["DDI", "Ja", "F1", "PRAUC", "Med"]
    test_table = PrettyTable(test_metric_header)
    test_table.align = "l"

    with open(model_path, "rb") as Fin:
        model.load_state_dict(torch.load(Fin, map_location=device))
    model = model.to(device).eval()

    logger.info("--------------------Begin Testing--------------------")
    ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
    tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)

    # np.random.seed(0)
    for _ in range(10):
        test_sample = np.random.choice(
            np.array(data_test, dtype=object), sample_size, replace=True
        )
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval_one_epoch(
            model, test_sample, voc_size, drug_data
        )

        logger.info(
            f"ddi_rate:{ddi_rate:.4f}, ja:{ja:.4f}, avg_f1:{avg_f1:.4f}, prauc:{prauc:.4f}"
        )
        result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)

    row_m = []
    for m, s in zip(mean, std):
        row_m.append(f"{m:.4f} ({s:.4f})")
    test_table.add_row(row_m)

    logger.info(f"\n{test_table.get_string()}")


def Train(
    model,
    device,
    data_train,
    data_eval,
    voc_size,
    drug_data,
    optimizer,
    log_dir,
    coef,
    target_ddi,
    EPOCH=50,
    wandb_logger=None,
):
    history, best_epoch, best_ja = defaultdict(list), 0, 0

    for epoch in range(EPOCH):
        model = model.train()
        tic, ddi_losses_epoch = time.time(), []
        for step, input_seq in enumerate(data_train):
            for adm_idx, adm in enumerate(input_seq):
                bce_target = torch.zeros((1, voc_size[2])).to(device)
                bce_target[:, adm[2]] = 1

                multi_target = -torch.ones((1, voc_size[2])).long()
                for idx, item in enumerate(adm[2]):
                    multi_target[0][idx] = item
                multi_target = multi_target.to(device)

                result, loss_ddi = model(
                    patient_data=input_seq[: adm_idx + 1], **drug_data
                )

                sigmoid_res = torch.sigmoid(result)

                loss_bce = binary_cross_entropy_with_logits(result, bce_target)
                loss_multi = multilabel_margin_loss(sigmoid_res, multi_target)

                result = sigmoid_res.detach().cpu().numpy()[0]
                result[result >= 0.5] = 1
                result[result < 0.5] = 0
                y_label = np.where(result == 1)[0]
                current_ddi_rate = ddi_rate_score(
                    [[y_label]], path="data/ddi_A_final.pkl"
                )

                if current_ddi_rate <= target_ddi:
                    loss = 0.95 * loss_bce + 0.05 * loss_multi
                else:
                    beta = coef * (1 - (current_ddi_rate / target_ddi))
                    beta = min(math.exp(beta), 1)
                    loss = (
                        beta * (0.95 * loss_bce + 0.05 * loss_multi)
                        + (1 - beta) * loss_ddi
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if wandb_logger is not None:
                    wandb_logger.log_metrics(
                        {
                            "train/loss_bce": loss_bce,
                            "train/loss_multi": loss_multi,
                        }
                    )

        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval_one_epoch(
            model, data_eval, voc_size, drug_data
        )

        if wandb_logger is not None:
            wandb_logger.log_metrics(
                {
                    "val/ddi": ddi_rate,
                    "val/ja": ja,
                    "val/prauc": prauc,
                    "val/f1": avg_f1,
                }
            )

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja
            logger.info(f"best ja in epoch {best_epoch}: {best_ja}")
            model.best_model_path = os.path.join(log_dir, "best_model.pth")
            torch.save(model.state_dict(), model.best_model_path)
