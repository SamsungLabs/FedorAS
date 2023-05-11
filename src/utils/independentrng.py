# Licensed under Apache 2.0 licence
# Created by:
#     * Javier Fernandez-Marques, Samsung AI Center, Cambridge
#     * Stefanos Laskaridis, Samsung AI Center, Cambridge
#     * Lukasz Dudziak, Samsung AI Center, Cambridge

import random

def get_rng_state(inc_torch=False, inc_numpy=False):
    state = {
        'random': random.getstate()
    }
    if inc_torch:
        import torch
        state['torch'] = torch.random.get_rng_state()
    if inc_numpy:
        import numpy as np
        state['numpy'] = np.random.get_state()

    return state


def set_rng_state(state):
    if not state:
        return
    random.setstate(state['random'])
    if 'torch' in state:
        import torch
        torch.random.set_rng_state(state['torch'])
    if 'numpy' in state:
        import numpy as np
        np.random.set_state(state['numpy'])


class IndependentRng():
    class PushRngState():
        def __init__(self, rng):
            self.rng = rng
            self.state = None

        def __enter__(self):
            self.state = self.rng.get_rng_state()
            set_rng_state(self.rng.state)
            return self.rng

        def __exit__(self, *args):
            self.rng.state = self.rng.get_rng_state()
            set_rng_state(self.state)

    def __init__(self, seed=None, inc_torch=False, inc_numpy=False):
        self.inc_numpy = inc_numpy
        self.inc_torch = inc_torch
        self.state = None
        with IndependentRng.PushRngState(self):
            if seed is not None:
                random.seed(seed)
                if inc_torch:
                    import torch
                    torch.manual_seed(seed)
                if inc_numpy:
                    import numpy as np
                    np.random.seed(seed)

    def activate(self):
        return IndependentRng.PushRngState(self)

    def get_rng_state(self):
        return get_rng_state(self.inc_torch, self.inc_numpy)
