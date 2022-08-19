from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
import argparse
import configparser


def main():

    config = configparser.ConfigParser()
    config.read('config.ini')
    subscription_id = config['DEFAULT']['AZURE_SUBSCRIPTION_ID']
    resource_group = config['DEFAULT']['AZURE_RESOURCE_GROUP']
    workspace_name = config['DEFAULT']['AZURE_WORKSPACE_NAME']
    ws = Workspace(subscription_id, resource_group, workspace_name)
    experiment = Experiment(workspace=ws, name='func-approx')

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=True, help='Number of CPUs/GPUs')
    parser.add_argument('-s', type=str, required=True, help='Script')
    parser.add_argument('-c', type=str, required=True, help='Container name')
    parser.add_argument('-cp', type=str, required=True, help='Cluster name')
    args = parser.parse_args()

    config = ScriptRunConfig(source_directory='.',
                             script=args.s,
                             arguments=['-c', args.c, '-d', './results'],
                             compute_target=args.cp)

    env = Environment.from_conda_specification(name='pytorch-env',
                                               file_path='env.yml')
    config.run_config.environment = env
    for i in range(args.n):
        print(f'Submitting job to CPU {i+1}...')
        run = experiment.submit(config)
        aml_url = run.get_portal_url()
        print(aml_url)


if __name__ == "__main__":
    main()
