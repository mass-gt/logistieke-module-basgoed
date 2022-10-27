import logging

import multiprocessing as mp
import numpy as np


def set_seed(
    seed: int,
    logger: logging.Logger,
    nameSeed: str = '',
    maxSeed: int = 1000000
) -> int:
    """
    Zet de random reeks vast.

    Args:
        seed (int): _description_
        logger (logging.Logger): _description_
        nameSeed (str, optional): _description_. Defaults to ''.
        maxSeed (int, optional): _description_. Defaults to 1000000.

    Returns:
        int: De toegepaste seed.
    """
    if seed is None:
        seed = np.random.randint(maxSeed)

        if nameSeed == '':
            logger.warning(f"Geen seed gedefinieerd, deze wordt nu op {seed} gezet.")
        else:
            logger.warning(
                f"Geen seed gedefinieerd voor '{nameSeed}', deze wordt nu op {seed} gezet.")

    np.random.seed(seed)

    return seed


def get_num_cpu(
    numCPU: int,
    maxCPU: int,
    logger: logging.Logger
):
    if numCPU is not None:
        try:
            numCPU = int(numCPU)

            if numCPU > mp.cpu_count():
                numCPU = max(1, min(mp.cpu_count() - 1, maxCPU))
                logger.warning(
                    "De waarde van de parameter '...-num-cpu' is te hoog. " +
                    f"Deze PC heeft slechts {mp.cpu_count()} processoren beschikbaar. " +
                    f"Daarom wordt '...-num-cpu' binnen de rekenmodule op {numCPU} gezet.")
            if numCPU < 1:
                numCPU = max(1, min(mp.cpu_count() - 1, maxCPU))

        except ValueError:
            numCPU = max(1, min(mp.cpu_count() - 1, maxCPU))
            logger.warning(
                "Kon de waarde van de parameter '...-num-cpu' niet naar een integer " +
                "converteren. " +
                f"Daarom wordt '...-num-cpu' binnen de rekenmodule op {numCPU} gezet.")

    else:
        numCPU = max(1, min(mp.cpu_count() - 1, maxCPU))

    return numCPU
