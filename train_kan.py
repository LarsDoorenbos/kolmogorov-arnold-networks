
import os
import yaml

import torch
import ignite.distributed as idist

import sys
import kan


def main(argv):
    params_file = "params.yml"

    with open(params_file, 'r') as f:
        params = yaml.safe_load(f)

    # Remove SLURM_JOBID to prevent ignite assume we are using SLURM to run multiple tasks.
    os.environ.pop("SLURM_JOBID", None)

    kan.train_kan(0, params)


if __name__ == "__main__":
    main(sys.argv)