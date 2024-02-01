import argparse
import logging
import os

import dill
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
from prettytable import PrettyTable
from torch.utils.tensorboard import SummaryWriter

from models import GAMENet, SafeDrugModel
from modules.gnn import graph_batch_from_smile
from modules.mymodel import MyModel
from util import buildPrjSmiles, ddi_rate_score, multi_label_metric

logger = logging.getLogger()
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Experiment For DrugRec")

    parser.add_argument("--ehr_adj_path", type=str)
    parser.add_argument("--ddi_adj_path", type=str)
    parser.add_argument("--Test", action="store_true", help="evaluating mode")

    parser.add_argument("--dim", default=64, type=int, help="model dimension")
    parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
    parser.add_argument("--dp", default=0.7, type=float, help="dropout ratio")
    parser.add_argument(
        "--resume_path",
        type=str,
        help="path of well trained model, only for evaluating the model",
    )
    parser.add_argument(
        "--device", type=int, default=0, help="gpu id to run on, negative for cpu"
    )
    parser.add_argument(
        "--target_ddi", type=float, default=0.06, help="expected ddi for training"
    )
    parser.add_argument(
        "--coef",
        default=2.5,
        type=float,
        help="coefficient for DDI Loss Weight Annealing",
    )

    parser.add_argument(
        "--epochs", default=50, type=int, help="the epochs for training"
    )
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--version", type=str)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--tf_writer", default=0, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    params = {"font.size": 8.0}
    plt.rcParams.update(params)
    keys = plt.rcParams.keys()

    seed = pl.seed_everything(661893727)
    args = parse_args()
    # logger
    log_dir = os.path.join(
        f"../saved/{args.model_name}/version-{args.version}/seed-{seed}"
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    casestudy_dir = os.path.join(f"../saved/casestudy/{args.model_name}/")
    args.casestudy_dir = casestudy_dir
    if not os.path.exists(casestudy_dir) and args.CaseStudy:
        os.makedirs(casestudy_dir)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(f"{log_dir}/run.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    #
    tf_writer = SummaryWriter(log_dir=log_dir) if args.tf_writer == 1 else None

    logger.info(f"Global seed set to {seed}")
    logger.info(f"Current PID: {os.getpid()}")
    logger.info(f"args:\n{args}")
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    data_path = "../data/records_final.pkl"
    voc_path = "../data/voc_final.pkl"
    ddi_adj_path = "../data/ddi_A_final.pkl"
    ddi_mask_path = "../data/ddi_mask_H.pkl"
    molecule_path = "../data/idx2SMILES.pkl"
    substruct_smile_path = "../data/substructure_smiles.pkl"
    with open(ddi_adj_path, "rb") as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, "rb") as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, "rb") as Fin:
        data = dill.load(Fin)
    with open(molecule_path, "rb") as Fin:
        molecule = dill.load(
            Fin
        )  # {'A01A': {'CC(=O)OC1=CC=CC=C1C(O)=O', '[F-].[Na+]', '[H][C@@]12C[C@@H](C)...'}
    with open(voc_path, "rb") as Fin:
        voc = dill.load(Fin)

    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]
    if args.debug == 1:
        data_train = data_train[:50]
        data_test = data_test[:50]
        data_eval = data_eval[:50]

    average_projection, smiles_list = buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)

    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {"batched_data": molecule_graphs.to(device)}
    molecule_para = {
        "num_layer": 4,
        "emb_dim": args.dim,
        "graph_pooling": "mean",
        "drop_ratio": args.dp,
        "gnn_type": "gin",
        "virtual_node": False,
    }

    with open(substruct_smile_path, "rb") as Fin:
        substruct_smiles_list = dill.load(Fin)

    substruct_graphs = graph_batch_from_smile(substruct_smiles_list)
    substruct_forward = {"batched_data": substruct_graphs.to(device)}
    substruct_para = {
        "num_layer": 4,
        "emb_dim": args.dim,
        "graph_pooling": "mean",
        "drop_ratio": args.dp,
        "gnn_type": "gin",
        "virtual_node": False,
    }

    model = MyModel(
        config=args,
        emb_dim=args.dim,
        voc_size=voc_size,
        substruct_num=ddi_mask_H.shape[1],
        device=device,
        dropout=args.dp,
    ).to(device)

    drug_data = {
        "substruct_data": substruct_forward,
        "mol_data": molecule_forward,
        "ddi_mask_H": ddi_mask_H,
        "tensor_ddi_adj": ddi_adj,
        "average_projection": average_projection,
    }

    model.load_state_dict(torch.load(open(args.resume_path, "rb"), map_location=device))
    model.to(device=device)
    model.eval()

    test_sample = data_test[686]
    test_adm_idx = 1
    ja_to_plot, F1_to_plot, prauc_to_plot = [], [], []
    # for test_adm_idx in range(len(test_sample)):
    test_metric_header = ["DDI", "Ja", "F1", "PRAUC", "Med"]
    test_table = PrettyTable(test_metric_header)

    result = []
    smm_record, ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(6)]
    med_cnt, visit_cnt = 0, 0

    y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []

    adm = test_sample[test_adm_idx]

    output, _ = model(patient_data=test_sample[: test_adm_idx + 1], **drug_data)

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

    ddi_rate = ddi_rate_score(
        smm_record, path="/home/zjj/code/MoleRec_cus/data/ddi_A_final.pkl"
    )

    ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = (
        ddi_rate,
        np.mean(ja),
        np.mean(prauc),
        np.mean(avg_p),
        np.mean(avg_r),
        np.mean(avg_f1),
        med_cnt / visit_cnt,
    )

    result.append([ddi_rate, ja, avg_f1, prauc, avg_med])

    result = np.array(result)
    mean, std = result.mean(axis=0), result.std(axis=0)

    row_m = []
    for m, s in zip(mean, std):
        row_m.append(f"{m:.4f} ({s:.4f})")
    test_table.add_row(row_m)

    print(f"\n{test_table.get_string()}")

    ja_to_plot.append(ja)
    F1_to_plot.append(avg_f1)
    prauc_to_plot.append(prauc)
    ###########plot weight
    dir = "/home/zjj/code/MoleRec_cus/saved/casestudy/MoleRecModel/"
    voc_path = "/home/zjj/code/MoleRec_cus/data/voc_final.pkl"
    with open(voc_path, "rb") as Fin:
        voc = dill.load(Fin)
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]

    diag_ICD = [diag_voc.idx2word[i] for i in adm[0]]
    proc_ICD = [pro_voc.idx2word[i] for i in adm[1]]
    print(f"diag_ICD: {diag_ICD}\nproc_ICD: {proc_ICD}")

    gt_med_idx = sorted(adm[2])
    pred_med_idx = sorted(y_pred_label_tmp)
    gt_med_atc = [med_voc.idx2word[idx] for idx in gt_med_idx]
    pred_med_atc = [med_voc.idx2word[idx] for idx in pred_med_idx]
    print(f"gt_med_idx: {sorted(gt_med_idx)}\npred_med_idx: {pred_med_idx}")
    print(f"gt_med_atc: {gt_med_atc}\npred_med_atc: {pred_med_atc}")
