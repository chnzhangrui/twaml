# -*- coding: utf-8 -*-

"""tW analysis machine learning: twaml

This python package contains modules for handling different machine
learning requirements for the ATLAS Full Run II tW analysis.

"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="{:20}  %(levelname)s  %(message)s".format("[twaml:%(name)s]"),
)

logging.addLevelName(
    logging.WARNING,
    "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)),
)
logging.addLevelName(
    logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR))
)
logging.addLevelName(
    logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO))
)
logging.addLevelName(
    logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG))
)


from .data import dataset
