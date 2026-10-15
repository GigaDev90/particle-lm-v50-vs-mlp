"""V50: V15 with half-sized initial resident write weights.

This initialization control begins with the same mathematical function as V48,
but uses V15's unscaled forward pass. It separates their common small initial
field from the subsequent training parameterization. It does not isolate only
learning rate: write gradients and shared clipping can also differ.
"""

import torch

if __package__:
    from . import v15 as reference
else:  # Frozen runs keep their exact reference dependency beside this module.
    import v15 as reference

ModelConfig = reference.ModelConfig
ParticleField = reference.ParticleField
Block = reference.Block
parameter_count = reference.parameter_count
config_dict = reference.config_dict


class GPT(reference.GPT):
    def __init__(self, config=None):
        super().__init__(config)
        # No extra random draws. This applies only during construction; loading
        # a checkpoint restores its saved tensors without scaling them again.
        with torch.no_grad():
            for block in self.blocks:
                block.field.down.weight.mul_(0.5)
