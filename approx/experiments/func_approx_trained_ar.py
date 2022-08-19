import os
import sys
import gc
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from approx.models.single_layer_simple import SingleLayerSimple
from approx.models.helper import ApproximationHelper
from approx.experiments.organizer import Organizer

logger = logging.getLogger(__name__)


def train(wid, func, results_folder, save_all_models=False):

    steps = 100
    n_train = 1000
    n_fine = 1000

    if func == '5gaussian':
        params = {'s': 0.2, 'g': 0.5}
        lr = 0.002
        intervals = 1000

    if func == '2gaussian':
        params = {'s': 0.2, 'g': 0.25}
        lr = 0.002
        intervals = 1000

    if func == '1gaussian':
        params = {'s': 0.2, 'g': 0.1}
        lr = 0.002
        intervals = 1000

    if func == 'cusp':
        params = None
        n_train += 1
        lr = 0.02
        intervals = 1000

    if func == 'step':
        params = None
        lr = 0.02
        intervals = 1000

    gamma = 0.2

    model = SingleLayerSimple(wid, ar_trained=True)
    writer = SummaryWriter(results_folder)
    helper = ApproximationHelper(func, n_train, n_fine, params=params)
    initial_br = model.l1.bias.data.clone()
    initial_ar = model.l2.weight.data.clone()

    x = helper.x
    y = helper.y

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=steps*intervals/4,
                                                gamma=gamma)

    df = pd.DataFrame(columns=['step', 'loss', 'res_l1', 'res_l2',
                               'res_H1', 'nn_H1',
                               'br_l1', 'br_l2', 'br_inf',
                               'ar_l1', 'ar_l2', 'ar_inf'])

    torch.save(model.state_dict(), os.path.join(results_folder, f'z0.pt'))

    for step in range(steps * intervals):

        optimizer.zero_grad()
        y_hat = model(x)

        loss = torch.nn.MSELoss()(y_hat, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % steps == 0:

            loss = loss.item()
            res_l1 = helper.residual_lp_norm(model, 1)
            res_l2 = helper.residual_lp_norm(model, 2)
            res_H1 = helper.residual_H1_norm(model)
            nn_H1 = helper.nn_H1_norm(model)

            # Distance from initial breakpoints in various norms
            br_diff = model.l1.bias.data - initial_br
            br_l1 = torch.norm(br_diff, 1).item() / wid
            br_l2 = torch.norm(br_diff, 2).item() / wid ** 0.5
            br_inf = torch.norm(br_diff, np.inf).item()

            ar_diff = model.l2.weight.data - initial_ar
            ar_l1 = torch.norm(ar_diff, 1).item() / wid
            ar_l2 = torch.norm(ar_diff, 2).item() / wid ** 0.5
            ar_inf = torch.norm(ar_diff, np.inf).item()

            df.loc[len(df.index)] = [step, loss, res_l1, res_l2,
                                     res_H1, nn_H1,
                                     br_l1, br_l2, br_inf,
                                     ar_l1, ar_l2, ar_inf]

            writer.add_scalar('Loss', loss, step)
            writer.add_scalar('res_l1', res_l1, step)
            writer.add_scalar('res_l2', res_l2, step)
            writer.add_scalar('res_H1', res_H1, step)
            writer.add_scalar('nn_H1', nn_H1, step)

            writer.add_scalar('br_l1', br_l1, step)
            writer.add_scalar('br_l2', br_l2, step)
            writer.add_scalar('br_inf', br_inf, step)

            writer.add_scalar('ar_l1', ar_l1, step)
            writer.add_scalar('ar_l2', ar_l2, step)
            writer.add_scalar('ar_inf', ar_inf, step)
            if save_all_models:
                torch.save(model.state_dict(), os.path.join(results_folder, f'z{step}.pt'))

    torch.save(model.state_dict(), os.path.join(results_folder, f'trained.pt'))
    writer.flush()
    writer.close()
    df.to_csv(os.path.join(results_folder, 'losses.csv'), index=False)


if __name__ == '__main__':

    logging.basicConfig(
        stream=sys.stdout,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    functions = ['step', 'cusp', '5gaussian', '2gaussian', '1gaussian']
    widths = [10, 18, 32, 56, 100, 178, 316, 562, 1000]
    seeds = list(range(100))

    conf = {'name': 'func-approx-trained-ar',
            'fixed_parameters': {},
            'variable_parameters': {'seed': seeds,
                                    'width': widths,
                                    'function': functions
                                    }
            }

    # TODO: add command-line args
    m_org = Organizer(config=conf, exp_folder='./results',
                      container_name='func-approx-trained-ar11')

    for test in m_org.run_tests():

        test_folder = m_org.get_test_folder(test)
        width = test['width']
        seed = test['seed']
        function = test['function']

        logger.info(test_folder)
        torch.manual_seed(seed)
        train(width, function, test_folder)
        m_org.complete_test()
        gc.collect()
