"""CPU checks for the published models and their original optimizer behavior.

Run from the repository root: python -m unittest discover -s tests -v
Optional export audit: set PARTICLE_LM_FROZEN_ROOT to a directory containing
v50/source and mlp/source from the original recorded runs. Those private
reference files are intentionally not required by this standalone package.
"""
import contextlib
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def import_bundle(directory):
    """Isolate the original sibling imports without changing either source."""
    names = ('model', 'train', 'v15')
    previous = {name: sys.modules.pop(name, None) for name in names}
    old_path = sys.path[:]
    sys.path.insert(0, str(directory))
    importlib.invalidate_caches()
    try:
        model = importlib.import_module('model')
        trainer = importlib.import_module('train')
        yield model, trainer
    finally:
        sys.path[:] = old_path
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


def toy_config(module, version):
    values = dict(vocab_size=32, context_length=12, n_layers=1,
                  width=16, n_heads=2, dropout=0.0)
    if version == 'v50':
        values.update(n_routing_heads=2, routing_dim=8, response_channels=2,
                      n_particles=8, radius_squared=2.171848773956299,
                      kernel_gain=6.617806196212754,
                      response_init_gain=1.0750019319619315)
    else:
        values['ff_width'] = 32
    return module.ModelConfig(**values)


class PublishedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.bundles = {}
        for version in ('v50', 'mlp'):
            with import_bundle(ROOT/'models'/version) as bundle:
                cls.bundles[version] = bundle

    def test_recorded_config_and_parameter_counts(self):
        for version, count in (('v50', 29_726_752), ('mlp', 29_676_544)):
            module, _ = self.bundles[version]
            config = json.loads((ROOT/'configs'/f'{version}.json').read_text())
            model = module.GPT(module.ModelConfig(**config))
            self.assertEqual(model.parameter_count(), count)
            self.assertIs(model.token_embedding.weight, model.lm_head.weight)
            self.assertEqual(model.config.width, 512)
            self.assertEqual(model.config.n_layers, 8)
            del model

    def test_v50_halves_only_initial_outputs_and_preserves_rng(self):
        module, _ = self.bundles['v50']
        config = toy_config(module, 'v50')
        torch.manual_seed(1337)
        parent = module.reference.GPT(config)
        parent_rng = torch.get_rng_state().clone()
        torch.manual_seed(1337)
        model = module.GPT(config)
        self.assertTrue(torch.equal(parent_rng, torch.get_rng_state()))
        for key, value in model.state_dict().items():
            expected = parent.state_dict()[key]
            if key.endswith('.field.down.weight'):
                expected = expected*0.5
            self.assertTrue(torch.equal(value, expected), key)

    def test_models_are_causal_and_do_not_change_input_tokens(self):
        for version, (module, _) in self.bundles.items():
            model = module.GPT(toy_config(module, version)).eval()
            tokens = torch.randint(0, 32, (2, 12))
            before = tokens.clone()
            changed = tokens.clone()
            changed[:, 6:] = (changed[:, 6:]+7) % 32
            with torch.no_grad():
                logits, loss = model(tokens, tokens)
                other, _ = model(changed, tokens)
            torch.testing.assert_close(logits[:, :6], other[:, :6], rtol=1e-7, atol=1e-7)
            self.assertTrue(torch.equal(tokens, before))
            self.assertTrue(torch.isfinite(loss))

    def test_v50_tangent_adam_step_matches_independent_formula(self):
        module, trainer = self.bundles['v50']
        model = module.GPT(toy_config(module, 'v50'))
        references = trainer.normalize_resident_positions(model)
        scale = 0.8671654817246768
        optimizer = trainer.make_optimizer(model, torch.device('cpu'), 3e-4, 0.1,
                                           scale, references)
        positions = model.blocks[0].field.raw_positions
        before = positions.detach().clone()
        gradient = torch.randn_like(positions)*0.2
        positions.grad = gradient.clone()
        beta1, beta2 = 0.9, 0.95
        m = (1-beta1)*gradient
        v = (1-beta2)*gradient.square()
        adaptive = (m/(1-beta1))/(v.sqrt()/(1-beta2)**0.5+1e-8)
        direction = torch.nn.functional.normalize(before, dim=-1)
        tangent = adaptive-(adaptive*direction).sum(-1, keepdim=True)*direction
        expected = torch.nn.functional.normalize(direction-3e-4*scale*tangent, dim=-1)
        optimizer.step()  # One synthetic CPU optimizer fixture, not a training run.
        torch.testing.assert_close(positions, expected, rtol=0, atol=1e-7)
        torch.testing.assert_close(positions.norm(dim=-1), torch.ones_like(positions[..., 0]),
                                   rtol=0, atol=2e-7)
        position_group = next(group for group in optimizer.param_groups if group['group_name'] == 'positions')
        self.assertEqual(position_group['weight_decay'], 0.0)
        self.assertEqual(position_group['lr'], 3e-4*scale)
        for name, parameter in model.named_parameters():
            if name in references:
                self.assertTrue(torch.equal(optimizer.state[parameter]['position_reference_norm'], references[name]))

    def test_v50_clipping_uses_saved_raw_norms_and_shared_coefficient(self):
        module, trainer = self.bundles['v50']
        model = module.GPT(toy_config(module, 'v50'))
        references = trainer.normalize_resident_positions(model)
        optimizer = trainer.make_optimizer(model, torch.device('cpu'), 3e-4, 0.1,
                                           0.8671654817246768, references)
        p = model.blocks[0].field.raw_positions
        down = model.blocks[0].field.down.weight
        p.grad = torch.randn_like(p)
        down.grad = torch.randn_like(down)
        old_p, old_down = p.grad.clone(), down.grad.clone()
        reference = optimizer.state[p]['position_reference_norm']
        expected_norm = torch.sqrt((old_p/reference).square().sum()+old_down.square().sum())
        coefficient = min(1.0, 0.2/(float(expected_norm)+1e-6))
        actual_norm = trainer.clip_gradients(model, optimizer, 0.2)
        torch.testing.assert_close(actual_norm, expected_norm, rtol=2e-7, atol=1e-6)
        torch.testing.assert_close(p.grad, old_p*coefficient, rtol=2e-7, atol=1e-7)
        torch.testing.assert_close(down.grad, old_down*coefficient, rtol=2e-7, atol=1e-7)

    def test_models_share_sampler_validation_offsets_and_lr_schedule(self):
        _, v50 = self.bundles['v50']
        _, mlp = self.bundles['mlp']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train', 'val'):
                (np.arange(4096, dtype=np.uint16) % 32).tofile(root/f'{name}.bin')
            data_a, data_b = (trainer.TokenData(root, 12, 1337) for trainer in (v50, mlp))
            np.testing.assert_array_equal(data_a.val_offsets, data_b.val_offsets)
            for _ in range(4):
                xa, ya = data_a.training_batch(3, torch.device('cpu'))
                xb, yb = data_b.training_batch(3, torch.device('cpu'))
                self.assertTrue(torch.equal(xa, xb) and torch.equal(ya, yb))
            self.assertEqual(data_a.sampler.bit_generator.state, data_b.sampler.bit_generator.state)
        for step in (0, 99, 100, 600, 15_260, 30_517):
            self.assertEqual(v50.cosine_lr(step, 30_518, 3e-4, 3e-5, 100),
                             mlp.cosine_lr(step, 30_518, 3e-4, 3e-5, 100))
        self.assertEqual(v50.EFFECTIVE_BATCH_TOKENS, 32_768)
        self.assertEqual(mlp.VALIDATION_TOKENS, 131_072)

    def test_export_hashes_and_portable_defaults(self):
        provenance = json.loads((ROOT/'docs/code-provenance.json').read_text())
        for record in provenance['files']:
            file = ROOT/record['public_file']
            self.assertEqual(hashlib.sha256(file.read_bytes()).hexdigest(), record['export_sha256'])
            if not record['changes']:
                self.assertEqual(record['original_sha256'], record['export_sha256'])
        _, trainer = self.bundles['v50']
        self.assertEqual(trainer.DEFAULT_DATA, 'data/fineweb-edu-1b-v1')
        self.assertFalse(hasattr(trainer, 'BASELINE_LOSS'))

    @unittest.skipUnless(os.environ.get('PARTICLE_LM_FROZEN_ROOT'),
                         'Optional original frozen sources are not bundled')
    def test_exported_vs_original_logits_gradients_rng_and_optimizer(self):
        reference_root = Path(os.environ['PARTICLE_LM_FROZEN_ROOT'])
        for version, (module, trainer) in self.bundles.items():
            with import_bundle(reference_root/version/'source') as (original_module, original_trainer):
                torch.manual_seed(47)
                original = original_module.GPT(toy_config(original_module, version))
                original_rng = torch.get_rng_state().clone()
                torch.manual_seed(47)
                exported = module.GPT(toy_config(module, version))
                self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
                for key, tensor in original.state_dict().items():
                    self.assertTrue(torch.equal(tensor, exported.state_dict()[key]), key)
                if version == 'v50':
                    old_refs = original_trainer.normalize_resident_positions(original)
                    new_refs = trainer.normalize_resident_positions(exported)
                    optimizer_old = original_trainer.make_optimizer(original, torch.device('cpu'), 3e-4, 0.1,
                                                                    0.8671654817246768, old_refs)
                    optimizer_new = trainer.make_optimizer(exported, torch.device('cpu'), 3e-4, 0.1,
                                                           0.8671654817246768, new_refs)
                else:
                    optimizer_old = original_trainer.make_optimizer(original, torch.device('cpu'), 3e-4, 0.1)
                    optimizer_new = trainer.make_optimizer(exported, torch.device('cpu'), 3e-4, 0.1)
                tokens = torch.randint(0, 32, (2, 12))
                old_logits, old_loss = original(tokens, tokens)
                new_logits, new_loss = exported(tokens, tokens)
                torch.testing.assert_close(new_logits, old_logits, rtol=0, atol=0)
                torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
                old_loss.backward()
                new_loss.backward()
                for (name, old_p), (new_name, new_p) in zip(original.named_parameters(), exported.named_parameters()):
                    self.assertEqual(name, new_name)
                    torch.testing.assert_close(new_p.grad, old_p.grad, rtol=0, atol=0)
                if version == 'v50':
                    original_trainer.clip_gradients(original, optimizer_old, 1.0)
                    trainer.clip_gradients(exported, optimizer_new, 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(original.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(exported.parameters(), 1.0)
                optimizer_old.step()
                optimizer_new.step()
                for name, old_tensor in original.state_dict().items():
                    self.assertTrue(torch.equal(old_tensor, exported.state_dict()[name]), name)


if __name__ == '__main__':
    unittest.main()
