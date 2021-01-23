from functools import partial
import itertools
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Relu, LogSoftmax, GeneralConv, Flatten, elementwise, parallel
from jax import jit, grad, lax, tree_map, random, tree_util
import jax.numpy as jnp

def piecewise_constant(boundaries, values, t):
    index = jnp.sum(boundaries < t)
    return jnp.take(values, index)


def create_stepped_learning_rate_fn(base_learning_rate, steps_per_epoch,
                                        lr_sched_steps, warmup_length=0.0):

    boundaries = [step[0] for step in lr_sched_steps]
    decays = [step[1] for step in lr_sched_steps]

    boundaries = jnp.array(boundaries) * steps_per_epoch
    boundaries = jnp.round(boundaries).astype(jnp.int32)
    values = jnp.array([1.0] + decays) * base_learning_rate

    def step_fn(step):
        lr = piecewise_constant(boundaries, values, step)
        if warmup_length > 0.0:
            lr = lr * jnp.minimum(1., step / float(warmup_length) / steps_per_epoch)

        return lr

    return step_fn

def huberLoss(x, delta: float = 1.0):
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)

    linear = abs_x - quadratic

    return 0.5 * quadratic ** 2 + delta * linear

class DDQN(object):
    def __init__(self, num_actions,
                    input_shape,
                    adam_params,
                    architecture,
                    mappingFunction = None,
                    seed = 0):

        self.seed = seed
        self.mappingFunction = mappingFunction

        rng = random.PRNGKey(self.seed)
        self.num_actions = num_actions
        self.input_shape = input_shape

        self.itercount = itertools.count()

        print("DDQN input shape: {}.".format(input_shape))

        initialize_params, self._predict = architecture

        rng1, rng2 = random.split(rng)

        _, self.Q1 = initialize_params(rng1, input_shape)
        _, self.Q2 = initialize_params(rng2, input_shape)

        self.predict = jit(self._predict)

        self._createOptimizers(adam_params)

    def _createOptimizers(self, params):
        print(params)
        learning_schedule = create_stepped_learning_rate_fn(
            base_learning_rate = params['step_size'],
            steps_per_epoch=1,
            lr_sched_steps=[[int(params['N_iterations'] / 8.0), 0.5]]
        )

        self.optimizer_init, self.optimizer_update, self.get_params = optimizers.adam(learning_schedule, b1=params['b1'], b2=params['b2'], eps=params['eps'])

        self.optimizer_state1 = self.optimizer_init(self.Q1)


    @partial(jit, static_argnums=(0,))
    def BellmanFunction(self, Qnet, batch, actions):
        inputs, targets = batch
        preds = self.predict(Qnet, inputs)

        if self.mappingFunction is not None:
            preds = self.mappingFunction(preds)

        predictions_select = jnp.take_along_axis(preds, jnp.expand_dims(actions, axis=1), axis=1)

        losses = huberLoss(predictions_select - targets)
        return jnp.mean(losses)


    @partial(jit, static_argnums=(0,))
    def _updateQNetwork(self, step, Qnet, opt_state, batch, actions):
        gradients = grad(self.BellmanFunction)(Qnet, batch, actions)

        clipped_gradients = tree_map(lambda g: jnp.clip(g, -10.0, 10.0), gradients)
        return self.optimizer_update(step, clipped_gradients, opt_state)


    @partial(jit, static_argnums=(0,))
    def DQlearning(self, Q1, Q2, gamma, rewards, next_states, is_terminals):
        nextQ1s = self.predict(Q1, next_states)
        nextQ2s = self.predict(Q2, next_states)

        if self.mappingFunction is not None:
            nextQ1s = self.mappingFunction(nextQ1s)
            nextQ2s = self.mappingFunction(nextQ2s)

        Q1MaxArgs = jnp.argmax(nextQ1s, axis=1)

        newQ2Max = jnp.take_along_axis(nextQ2s, jnp.expand_dims(Q1MaxArgs, axis=1), axis=1)[:, 0]

        Q_values = rewards + gamma * newQ2Max * (1 - is_terminals)

        Q_values = lax.stop_gradient(Q_values)

        return Q_values


    def update(self, replayBuffer, batchSize, gamma):
        states, actions, rewards, nextStates, isTerminals = replayBuffer.sample(batchSize)

        Q = self.Q1
        Q_Prime = self.Q2

        Q_values = self.DQlearning(Q, Q_Prime, gamma, rewards, nextStates, isTerminals)
    
        self.optimizer_state1 = self._updateQNetwork(next(self.itercount), 
                                                          self.get_params(self.optimizer_state1), 
                                                          self.optimizer_state1, 
                                                          [states, Q_values[:, None]],
                                                          actions)
        self.Q1 = self.get_params(self.optimizer_state1)

    def updateTarget(self):
        self.Q2 = self.get_params(self.optimizer_state1)


