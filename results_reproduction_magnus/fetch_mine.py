# Copy of the authors' results/fetch_wandb_training.py with four changes:
#   1. only runs started by --user are fetched;
#   2. the classifier config must match exactly (<name>.json), not as a substring;
#   3. runs that logged no data (crashed/killed) are skipped;
#   4. --group X fetches only runs in W&B group/tag X; without it, runs with any group/tag are excluded.
import argparse
import os
import pickle

import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_name", type=str, help="Wandb entity (team) name.")
    parser.add_argument("--user", type=str, default="nxp334", help="Only fetch runs started by this W&B user.")
    parser.add_argument("--dataset", type=str, help="Dataset name.")
    parser.add_argument("--batch_size", type=int, help="Labels collected per round.")
    parser.add_argument("--embed_model", type=str, help="Substring of the embedding model name.")
    parser.add_argument("--classifier_model", type=str, help="Classifier model config name, without .json.")
    parser.add_argument("--trainer_config", type=str, help="Substring of the trainer name.")
    parser.add_argument("--corrupter_config", type=str, default="noiseless", help="Substring of the corrupter name.")
    parser.add_argument("--group", type=str, default=None,
                        help="W&B group/tag to fetch (e.g. randinit); omit for runs without any group/tag.")
    args = parser.parse_args()

    corrupter_str = "" if "noiseless" in args.corrupter_config else "_%s" % args.corrupter_config
    path_name = "./%s%s/bs_%d/embed_%s/model_%s/trainer_%s" % (
        args.dataset, corrupter_str, args.batch_size, args.embed_model, args.classifier_model, args.trainer_config)

    written = set()
    api = wandb.Api(timeout=100)
    for run in api.runs("%s/Active Learning, %s, Batch Size=%d" % (args.wandb_name, args.dataset, args.batch_size)):
        config = run.config
        labels = {run.group, *run.tags} - {None, ""}
        wanted = (args.group in labels) if args.group else not labels
        if run.user.username != args.user or not wanted:
            continue
        if not ((args.embed_model in config["embed_model_config"]) and
                (config["classifier_model_config"] == args.classifier_model + ".json") and
                (args.trainer_config in config["trainer_config"]) and
                (("corrupter_config" not in config and "noiseless" in args.corrupter_config) or
                 ("corrupter_config" in config and args.corrupter_config in config["corrupter_config"]))):
            continue

        run_dict, label_dict = {}, {}
        for row in run.scan_history():
            if row.get("Num Labeled") is not None:
                for key in row:
                    run_dict.setdefault(key, []).append(row[key])
            if row.get("i") is not None:
                label_dict[row["i"]] = row["Label index"]
        if "Num Labeled" not in run_dict:
            continue
        label_lst = [label_dict.get(i) for i in range(len(label_dict))]
        print(run.name, config["seed"], len(run_dict["Num Labeled"]), "rounds")

        file_name = "%s/%s_%d.pkl" % (path_name, config["strategy_config"].split(".")[0], config["seed"])
        if file_name in written:
            print("  warning: a second run with this strategy and seed overwrites", file_name)
        written.add(file_name)
        os.makedirs(path_name, exist_ok=True)
        with open(file_name, "wb") as file:
            pickle.dump([run_dict, label_lst], file)
