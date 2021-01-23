import numpy as np
import jax 
import jax.numpy as jnp

def calculateEpsilon(iteration, epsilonThresholds, epsilonValues): 
   
    for i in range(len(epsilonThresholds) - 1):
        if epsilonThresholds[i] <= iteration and iteration < epsilonThresholds[i + 1]:
            alpha = float(iteration - epsilonThresholds[i]) / (epsilonThresholds[i + 1] - epsilonThresholds[i])
            return epsilonValues[i] * (1 - alpha) + alpha * epsilonValues[i + 1]
                       
    return epsilonValues[-1]

def chooseGreedyAction(model, state):
    Qnet = model.Q1
    Qvalues = model.predict(Qnet, np.array(state).reshape((1,) + model.input_shape[1:]))

    if model.mappingFunction is not None:
        Qvalues = model.mappingFunction(Qvalues)

    Qvalues = Qvalues.reshape(-1)

    return jnp.argmax(Qvalues)

def takeEpisodeStep(iteration,
                      environment, Qnet, buffer,
                      currentIndex, state,
                      epsilonSchedule = None):

    epsilon = calculateEpsilon(iteration, **epsilonSchedule)

    randNum = np.random.uniform()
    if randNum < epsilon:
        action = environment.action_space.sample()

    else:
        action = chooseGreedyAction(Qnet, state)

    frame, reward, isTerminal, _ = environment.step(action)
    nextState = frame

    buffer.recordEffect(currentIndex, action, reward, isTerminal)

    return nextState, reward, isTerminal

