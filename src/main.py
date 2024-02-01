import argparse
import logging
import os
from datetime import datetime

import dill
import pytorch_lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from torch.optim import Adam

from modules.gnn import graph_batch_from_smile
from modules.mymodel import MyModel
from training import Test, Train
from util import buildPrjSmiles, set_logger

logger = logging.getLogger()


def parse_args():
    parser = argparse.ArgumentParser("Experiment For DrugRec")
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
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    return args


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        logger.info(arg + "." * (str_num - len(arg) - len(str(val))) + str(val))


def main():
    args = parse_args()

    log_dir = f"run_logs/{args.model_name}/ver-{args.version}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    set_logger(log_dir)
    print_args(args)

    wandb_logger = (
        WandbLogger(
            project="CSRec",
            group=f"{args.model_name}",
            job_type=f"{args.version}",
            mode="disabled" if args.dev else "online",
        )
        if args.wandb
        else None
    )
    seed = pl.seed_everything()

    logger.info(f"Current PID: {os.getpid()}")
    logger.info(f"Global seed set to: {seed}")
    logger.info(f"CWD:{os.getcwd()}")
    # data
    if not torch.cuda.is_available() or args.device < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    data_path = "data/records_final.pkl"
    voc_path = "data/voc_final.pkl"
    ddi_adj_path = "data/ddi_A_final.pkl"
    ddi_mask_path = "data/ddi_mask_H.pkl"
    molecule_path = "data/idx2SMILES.pkl"
    substruct_smile_path = "data/substructure_smiles.pkl"
    with open(ddi_adj_path, "rb") as Fin:
        ddi_adj = torch.from_numpy(dill.load(Fin)).to(device)
    with open(ddi_mask_path, "rb") as Fin:
        ddi_mask_H = torch.from_numpy(dill.load(Fin)).to(device)
    with open(data_path, "rb") as Fin:
        data = dill.load(Fin)
    with open(molecule_path, "rb") as Fin:
        molecule = dill.load(Fin)
    with open(voc_path, "rb") as Fin:
        voc = dill.load(Fin)

    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point : split_point + eval_len]
    data_eval = data[split_point + eval_len :]
    if args.dev:
        data_train = data_train[:50]
        data_test = data_test[:50]
        data_eval = data_eval[:50]

    average_projection, smiles_list = buildPrjSmiles(molecule, med_voc.idx2word)
    average_projection = average_projection.to(device)

    molecule_graphs = graph_batch_from_smile(smiles_list)
    molecule_forward = {"batched_data": molecule_graphs.to(device)}

    with open(substruct_smile_path, "rb") as Fin:
        substruct_smiles_list = dill.load(Fin)

    substruct_graphs = graph_batch_from_smile(substruct_smiles_list)
    substruct_forward = {"batched_data": substruct_graphs.to(device)}

    drug_data = {
        "substruct_data": substruct_forward,
        "mol_data": molecule_forward,
        "ddi_mask_H": ddi_mask_H,
        "tensor_ddi_adj": ddi_adj,
        "average_projection": average_projection,
    }
    # model
    model = MyModel(
        args=args,
        emb_dim=args.dim,
        voc_size=voc_size,
        substruct_num=ddi_mask_H.shape[1],
        ehr_adj_path="/home/zjj/code/KE-HMFNet/data/ehr_adj_final.pkl",
        ddi_adj_path="/home/zjj/code/KE-HMFNet/data/ddi_A_final.pkl",
        device=device,
        dropout=args.dp,
    ).to(device)

    if args.Test:
        Test(model, args.resume_path, device, data_test, voc_size, drug_data)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
        Train(
            model,
            device,
            data_train,
            data_eval,
            voc_size,
            drug_data,
            optimizer,
            log_dir,
            args.coef,
            args.target_ddi,
            EPOCH=args.epochs,
            wandb_logger=wandb_logger,
        )

        resume_path = model.best_model_path
        Test(model, resume_path, device, data_test, voc_size, drug_data)


if __name__ == "__main__":
    main()
