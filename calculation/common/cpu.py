import multiprocessing as mp


def determine_num_cpu(
    varDict: dict,
    maxCPU: int = 16
):
    if varDict['N_CPU'] not in ['', '""', "''"]:
        try:
            nCPU = int(varDict['N_CPU'])

            if nCPU > mp.cpu_count():
                nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))
                print(
                    'N_CPU parameter too high. Only ' +
                    str(mp.cpu_count()) +
                    ' CPUs available. Hence defaulting to ' +
                    str(nCPU) +
                    'CPUs.')

            if nCPU < 1:
                nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))

        except ValueError:
            nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))
            print(
                'Could not convert N_CPU parameter to an integer. ' +
                'Hence defaulting to ' + str(nCPU) + 'CPUs.')

    else:
        nCPU = max(1, min(mp.cpu_count() - 1, maxCPU))

    return nCPU
