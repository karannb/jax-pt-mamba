from typing import Any, Dict
from collections.abc import Callable

import optax

import jax
from flax import core, struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT


class DualOptTrainState(struct.PyTreeNode):
    """
    Need two optimizers to fully simulate the original scheduler used in torch.
    """
    step: int | jax.Array
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx1: optax.GradientTransformation = struct.field(pytree_node=False)
    tx2: optax.GradientTransformation = struct.field(pytree_node=False)
    opt1_state: optax.OptState = struct.field(pytree_node=True)
    opt2_state: optax.OptState = struct.field(pytree_node=True)
    batch_stats: Dict[str, Any]
    
    def apply_gradients(self, *, grads, **kwargs):
        """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

        Note that internally this function calls ``.tx.update()`` followed by a call
        to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

        Args:
        grads: Gradients that have the same pytree structure as ``.params``.
        **kwargs: Additional dataclass attributes that should be ``.replace()``-ed.

        Returns:
        An updated instance of ``self`` with ``step`` incremented by one, ``params``
        and ``opt_state`` updated by applying ``grads``, and additional attributes
        replaced as specified by ``kwargs``.
        """
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads['params']
            params_with_opt = self.params['params']
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        updates1, new_opt1_state = self.tx1.update(
            grads_with_opt, self.opt1_state, params_with_opt
        )
        intermediate_params_with_opt = optax.apply_updates(params_with_opt, updates1)
        
        updates2, new_opt2_state = self.tx2.update(
            grads_with_opt, self.opt2_state, intermediate_params_with_opt
        )
        new_params_with_opt = optax.apply_updates(intermediate_params_with_opt, updates2)

        # As implied by the OWG name, the gradients are used directly to update the
        # parameters.
        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                'params': new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
            
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt1_state=new_opt1_state,
            opt2_state=new_opt2_state,
            **kwargs,
        )
    
    @classmethod
    def create(cls, *, apply_fn, params, tx1, tx2, **kwargs):
        """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
        # We exclude OWG params when present because they do not need opt states.
        params_with_opt = (
            params['params'] if OVERWRITE_WITH_GRADIENT in params else params
        )
        opt1_state = tx1.init(params_with_opt)
        opt2_state = tx2.init(params_with_opt)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx1=tx1,
            tx2=tx2,
            opt1_state=opt1_state,
            opt2_state=opt2_state,
            **kwargs,
        )