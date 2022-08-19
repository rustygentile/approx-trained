import os
import sys
import itertools
import copy
import json
import logging
import configparser
import datetime as dt
import uuid
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError


logger = logging.getLogger(__name__)


class Organizer:
    """
    Handles organizing test results into subfolders and syncing experiments
    that run on multiple CPU's
    """
    def __init__(self, config, exp_folder='results',
                 container_name=None, timeout=3600):
        self.config = config
        self.exp_folder = exp_folder
        self.status_file = None
        self.status_msg = None
        self.timeout = timeout

        if container_name is not None:
            self.cloud = True
            self.container_name = container_name

            config = configparser.ConfigParser()
            config.read('config.ini')
            connection_string = \
                config['DEFAULT']['AZURE_STORAGE_CONNECTION_STRING']

            self.blob_service_client = BlobServiceClient.\
                from_connection_string(connection_string)

            try:
                self.blob_service_client.create_container(self.container_name)
            except ResourceExistsError:
                logger.info(f'{container_name} already exists...')

            logging.getLogger(
                'azure.core.pipeline.policies.http_logging_policy'
            ).setLevel(logging.WARNING)
        else:
            self.cloud = False

    def get_variable_vars(self):
        var_names = self.config['variable_parameters'].keys()
        var_values = [self.config['variable_parameters'][k] for k in var_names]
        return var_names, var_values

    def get_test_folder(self, test_config):
        var_names, _ = self.get_variable_vars()
        test_folder = '-'.join([v[0] + str(test_config[v]) for v in var_names])
        return os.path.join(self.exp_folder, 'tests', test_folder)

    def gen_test_configs(self):
        # Copy all the fixed parameters to a new config
        base_config = {'name': self.config['name']}
        for k in self.config['fixed_parameters'].keys():
            base_config[k] = self.config['fixed_parameters'][k]

        # Yields a full-factorial DOE
        counter = 0
        var_names, var_values = self.get_variable_vars()
        for product in itertools.product(*var_values):
            config = copy.deepcopy(base_config)
            for idx, v in enumerate(var_names):
                config[v] = product[idx]

            config['_test'] = counter
            counter += 1
            yield config

    def start_test(self, config):

        my_dir = self.get_test_folder(config)
        test_num = config['_test']

        if not os.path.exists(my_dir):
            os.makedirs(my_dir)

        self.status_file = os.path.join(my_dir, 'status.txt')

        if self.cloud:
            try:
                test_folder = os.path.dirname(self.status_file)
                blob_folder = 'tests/' + os.path.basename(test_folder)
                blob_client = self.blob_service_client.\
                    get_blob_client(container=self.container_name,
                                    blob=blob_folder + '/' + 'status.txt')
                with open(self.status_file, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
            except ResourceNotFoundError:
                pass

        try:
            with open(self.status_file, 'r') as f:
                status = f.read()
                if status == 'Complete!':
                    logger.info(f'{my_dir}, test {test_num} already complete!')
                    return False
                elif status.startswith('In progress... '):
                    ts1 = int(dt.datetime.utcnow().timestamp())
                    ts2 = int(status.split(': ')[1])
                    if abs(ts1 - ts2) > self.timeout:
                        logger.info(
                            f'{my_dir}, test {test_num} timeout... restarting')
                    else:
                        logger.info(
                            f'{my_dir}, test {test_num} already in progress... '
                            f'skipping for now')
                        return False
                else:
                    print(status)

        except FileNotFoundError:
            pass

        ts = int(dt.datetime.utcnow().timestamp())
        self.status_msg = f'In progress... {str(uuid.uuid4())} ' \
                          f'Job started: {ts} '
        with open(self.status_file, 'w+') as f:
            f.write(self.status_msg)

        if self.cloud:
            blob_name = blob_folder + '/' + 'status.txt'
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name)

            with open(self.status_file, 'rb') as data:
                blob_client.upload_blob(data, overwrite=True)

        config_file = os.path.join(my_dir, 'config.json')
        with open(config_file, 'w') as f:
            f.write(json.dumps(config, indent=4))

        return True

    def complete_test(self):

        with open(self.status_file, 'w') as f:
            f.write('Complete!')

        if self.cloud:
            test_folder = os.path.dirname(self.status_file)
            blob_folder = 'tests/' + os.path.basename(test_folder)

            test_folder = os.path.dirname(self.status_file)
            blob_client = self.blob_service_client.\
                get_blob_client(container=self.container_name,
                                blob=blob_folder + '/' + 'status.txt')

            cloud_status_msg = blob_client.download_blob().readall()\
                .decode("utf-8")

            if cloud_status_msg != self.status_msg:
                logger.info(
                    'Job completed by another cluster! Skipping upload...'
                )
                return False

            for f in os.listdir(test_folder):
                ow = 'status.txt' in f
                blob_name = blob_folder + '/' + f
                blob_client = self.blob_service_client.\
                    get_blob_client(container=self.container_name,
                                    blob=blob_name)

                with open(os.path.join(test_folder, f), 'rb') as data:
                    blob_client.upload_blob(data, overwrite=ow)

        return True

    def run_tests(self):

        my_dir = self.exp_folder
        if not os.path.exists(my_dir):
            os.makedirs(my_dir)

        script_name = os.path.basename(os.path.realpath(sys.argv[0]))
        script_path = os.path.dirname(os.path.realpath(sys.argv[0]))

        with open(os.path.join(script_path, script_name), 'r') as f:
            script_contents = f.read()

        with open(os.path.join(self.exp_folder, script_name), 'w') as f:
            f.write(script_contents)

        with open(os.path.join(self.exp_folder, 'args.txt'), 'w') as f:
            f.write('\n'.join(sys.argv))

        with open(os.path.join(self.exp_folder, 'config.json'), 'w') as f:
            f.write(json.dumps(self.config, indent=4))

        if self.cloud:
            files = [f for f in os.listdir(self.exp_folder) if
                     os.path.isfile(os.path.join(self.exp_folder, f))]
            for f in files:
                blob_client = self.blob_service_client.\
                    get_blob_client(container=self.container_name, blob=f)
                with open(os.path.join(self.exp_folder, f), 'rb') as data:
                    blob_client.upload_blob(data, overwrite=True)

        for my_conf in self.gen_test_configs():
            if self.start_test(my_conf):
                yield my_conf
