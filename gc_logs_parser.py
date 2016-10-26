import os
import subprocess


def download_last_x_logs(x):
    """
    Downloads last X log-files
    :param x: if x='all' => downloads all logs
              else - download x last log files
    :return: list of strings - log files addresses
    """
    command = 'gsutil'
    page = 'gs://fzz_logs'
    address = "/home/developer/logs"
    os.chdir(address)
    if x == 'all':
        last_x_log = subprocess.check_output([command, 'ls ' + page]).split('\n')[:-1]
    else:
        last_x_log = subprocess.check_output([command, 'ls ' + page]).split('\n')[-(x + 1):-1]
    saved_logs = []
    for log in last_x_log:
        filename = log[len(page) + 1:] + ".csv"
        subprocess.call([command, 'cp ' + log + ' ' + filename])
        saved_logs.append(address + '/' + filename)
    return saved_logs
