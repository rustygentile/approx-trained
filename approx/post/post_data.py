import sys
import csv
import json
import pandas as pd
import numpy as np
import argparse
import logging
import configparser
import os
from approx.experiments.organizer import Organizer

logger = logging.getLogger(__name__)


def gen_command(container_name, destination='.', pattern='*.csv'):

    config = configparser.ConfigParser()
    config.read('config.ini')
    connection_string = config['DEFAULT']['AZURE_STORAGE_CONNECTION_STRING']
    azure_cli_cmd = config['DEFAULT']['AZURE_CLI_CMD']
    azure_storage_account = config['DEFAULT']['AZURE_STORAGE_ACCOUNT']

    container_url = f'https://{azure_storage_account}' + \
                    f'.blob.core.windows.net/{container_name}'

    args = f'--source {container_url} ' + \
           f'--connection-string "{connection_string}" ' + \
           f'--pattern {pattern} ' + \
           f'--destination {destination} '

    command = f'{azure_cli_cmd} storage blob download-batch {args}'
    return command


def get_blob(container_name, destination='.', pattern='*.csv'):
    os.system(gen_command(container_name,
                          destination=destination,
                          pattern=pattern))


def main(data_path, container_name=None):

    with open(os.path.join(data_path, 'config.json'), 'r') as f:
        conf = json.loads(f.read())

    m_org = Organizer(config=conf, exp_folder=data_path)
    with open(os.path.join(data_path, 'results.csv'), 'w', newline='') as f:

        loss_vars = ['loss', 'res_l1', 'res_l2', 'res_H1', 'nn_H1',
                     'br_l1', 'br_l2', 'br_inf',
                     'ar_l1', 'ar_l2', 'ar_inf',
                     'res_H1_max', 'nn_H1_max']

        writer = csv.writer(f, delimiter=',')
        writer.writerow(['width', 'func', 'seed'] + loss_vars)

        for test in m_org.gen_test_configs():

            test_folder = m_org.get_test_folder(test)
            width = test['width']
            func = test['function']
            seed = test['seed']

            try:
                df = pd.read_csv(os.path.join(test_folder, 'losses.csv'))

                vals = []
                for v in loss_vars:
                    if '_max' not in v:
                        vals.append(df[v].values[-1])
                    else:
                        vals.append(max(df[v.replace('_max', '')].values))

            except FileNotFoundError:
                vals = [np.nan for v in loss_vars]

            writer.writerow([width, func, seed] + vals)


if __name__ == '__main__':

    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, required=False, help='Container name')
    parser.add_argument('-d', type=str, required=True, help='Destination folder')

    args = parser.parse_args()

    if args.c is not None:
        get_blob(args.c, args.d, '*.csv')
        get_blob(args.c, args.d, '*.json')
        get_blob(args.c, args.d, '*tfevents*')
        get_blob(args.c, args.d, '*.pt')

    main(args.d, args.c)
