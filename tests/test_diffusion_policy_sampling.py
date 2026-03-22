import torch

from vitra.models.action_model.diffusion_policy import DiffusionPolicy


class _DummyDiffusion:
    def __init__(self):
        self.calls = 0

    def p_sample_loop(self, sample_fn, shape, noise, **kwargs):
        self.calls += 1
        # Exercise sample function once to mimic sampling invocation semantics.
        kwargs["model_kwargs"]["z"]
        return noise


class _DummyNet:
    def forward(self, x, t, z, state=None, state_mask=None):
        return x[..., :2]



def test_sample_non_ddim_first_call_uses_base_diffusion():
    policy = DiffusionPolicy.__new__(DiffusionPolicy)
    policy.future_action_window_size = 2
    policy.in_channels = 2
    policy.use_state = None
    policy.diffusion = _DummyDiffusion()
    policy.ddim_diffusion = None
    policy.net = _DummyNet()

    action_features = torch.randn(1, 1, 4)
    action_masks = torch.ones(1, 3, 2)

    samples = policy.sample(
        action_features=action_features,
        cfg_scale=1.0,
        current_state=None,
        current_state_mask=None,
        use_ddim=False,
        num_ddim_steps=None,
        action_masks=action_masks,
    )

    assert samples.shape == (1, 3, 2)
    assert policy.diffusion.calls == 1
