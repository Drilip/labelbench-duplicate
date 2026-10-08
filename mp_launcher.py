import subprocess
import os
import argparse
import shlex
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_name", type=str, help="Wandb user name.")
parser.add_argument("--dataset", type=str, help="Dataset name.")
parser.add_argument("--data_dir", type=str, help="Directory to store dataset.", default="./data")
parser.add_argument("--metric", type=str, help="Metric name.")
parser.add_argument("--batch_size", type=int,
                    help="We train neural network after collecting batch_size number of new labels.")
parser.add_argument("--num_batch", type=int, help="Number of batches of collected labels.")
parser.add_argument("--embed_model_config", type=str, help="Path to the encoder model configuration file.",
                    default="none.json")
parser.add_argument("--classifier_model_config", type=str, help="Path to model configuration file.")
parser.add_argument("--trainer_config", type=str, help="Path to trainer configuration file.")
parser.add_argument("--corrupter_config", type=str, help="Path to corrupter configuration file.",
                    default="noiseless.json")
parser.add_argument("--strategies", type=str, nargs="+", help="Path to AL strategy configuration files.")
parser.add_argument("--num_runs", type=int, default=4, help="Number of runs.")
parser.add_argument("--run_per_device", type=int, default=1, help="Number of runs on each GPU.")
parser.add_argument("--device_per_run", type=int, default=0, help="Number of GPUs for each run.")
parser.add_argument("--gpu_masks", type=int, nargs="+", help="Which devices to run on.")
parser.add_argument("--skip", type=int, default=0, help="Skip the specified number of experiments in the beginning.")
parser.add_argument("--dry_run", action="store_true",
                    help="Print every generated child command without starting it.")
args = parser.parse_args()

python_dir = sys.executable


def wait_for_processes(running_processes):
    failures = []
    for process in running_processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((return_code, process.args))
    if failures:
        details = "\n".join(
            f"exit {return_code}: {shlex.join(command)}" for return_code, command in failures
        )
        raise RuntimeError(f"One or more training jobs failed:\n{details}")


counter = 0
processes = []
for strategy in args.strategies:
    if args.num_runs == 1:
        seeds = [1234]
    else:
        seeds = [1234 + (9999999 - 1234) * i // (args.num_runs - 1) for i in range(args.num_runs)]
    for i, seed in enumerate(seeds):
        command = [python_dir, "main.py",
                   "--seed", str(seed),
                   "--wandb_name", args.wandb_name,
                   "--dataset", args.dataset,
                   "--data_dir", args.data_dir,
                   "--metric", args.metric,
                   "--batch_size", str(args.batch_size),
                   "--num_batch", str(args.num_batch),
                   "--embed_model_config", args.embed_model_config,
                   "--classifier_model_config", args.classifier_model_config,
                   "--strategy_config", strategy,
                   "--trainer_config", args.trainer_config,
                   "--corrupter_config", args.corrupter_config
                   ]
        if counter >= args.skip:
            new_env = os.environ.copy()
            if args.dry_run:
                print(shlex.join(command))
            elif args.device_per_run > 0:
                num_parallel_runs = len(args.gpu_masks) // args.device_per_run
                assert num_parallel_runs > 0
                new_env["CUDA_VISIBLE_DEVICES"] = \
                    "".join([str(counter % num_parallel_runs * args.device_per_run + i) + "," for i in
                             range(args.device_per_run)])[:-1]
                processes.append(subprocess.Popen(command, env=new_env))
                if len(processes) == num_parallel_runs:
                    wait_for_processes(processes)
                    processes = []
            else:
                new_env["CUDA_VISIBLE_DEVICES"] = str(
                    args.gpu_masks[(counter % (len(args.gpu_masks) * args.run_per_device)) // args.run_per_device])
                processes.append(subprocess.Popen(command, env=new_env))
                if len(processes) == len(args.gpu_masks) * args.run_per_device:
                    wait_for_processes(processes)
                    processes = []
        counter += 1

wait_for_processes(processes)
