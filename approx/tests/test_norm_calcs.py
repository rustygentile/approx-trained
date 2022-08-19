import pytest
import torch
from approx.models.helper import ApproximationHelper
from approx.models.single_layer_simple import SingleLayerSimple


class TestNorms(object):

    @pytest.mark.unit
    def test_lp_norm(self):

        model = SingleLayerSimple(100)
        print('')
        print('func, m, ||phi - f||_2')
        print('----------------------')
        for tf in ['step', 'cusp', 'gaussian']:
            for nf in [1e3, 1e4, 1e5, 1e6]:
                helper = ApproximationHelper(tf, 100, int(nf))
                norm = helper.residual_lp_norm(model, 2)
                print(f'{tf}, {nf}, {norm}')

    @pytest.mark.unit
    def test_H1_norm(self):
        """
        Note: step and cusp functions are not in H^1
        """
        model = SingleLayerSimple(100)
        print('')
        print('func, m, ||phi - f||_{H^1}')
        print('--------------------------')
        for tf in ['step', 'cusp', 'gaussian']:
            for nf in [1e3, 1e4, 1e5, 1e6]:
                helper = ApproximationHelper(tf, 100, int(nf))
                norm = helper.residual_H1_norm(model)
                print(f'{tf}, {nf}, {norm}')

    @pytest.mark.unit
    def test_optimal_step(self):

        for width in range(50, 1000, 50):

            # Create a model with all breakpoints at +/- epsilon
            model = SingleLayerSimple(width)
            pb, mb = model.get_breakpoints()

            ipb = torch.nonzero(torch.isnan(pb.flatten())).detach().numpy().flatten()
            imb = torch.nonzero(torch.isnan(mb.flatten())).detach().numpy().flatten()

            n_pts = 2 * (min([len(ipb), len(imb)]) // 2)
            epsilon = 0.5 / n_pts / model.gain
            new_biases = -1 * torch.ones(width, dtype=torch.float32)

            for i in range(n_pts):
                new_biases[ipb[i]] = -epsilon
                new_biases[imb[i]] = epsilon

            model.l1.bias.data = new_biases

            helper = ApproximationHelper('step', 100, int(width * 100))
            norm = helper.residual_lp_norm(model, 1)
            error = norm - 0.5 * epsilon
            print(f'm: {width} l1: {norm} eps: {epsilon} err: {error}')

            # Expect the l1 norm to be 1/2 epsilon here
            assert abs(error) < 1e-3
