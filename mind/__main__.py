import sys
import logging

from .run import train, test
from .parameters import parse_args
from pathlib import Path
from .utils import setuplogger

sys.path.append('..')


args = parse_args()
Path(args.model_dir).mkdir(parents=True, exist_ok=True)
setuplogger(args.log_file)
logging.info(args)
if "train" in args.mode:
    train(args)
if "test" in args.mode:
    test(args)