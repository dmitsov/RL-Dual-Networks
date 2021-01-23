import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax, GeneralConv, Flatten, elementwise, parallel, FanOut, FanInConcat
from jax import jit
import jax.numpy as jnp
import numpy as np
import random
import os, time
import AtariWrappers
from DDQN import DDQN
from Logger import Logger
from ReplayBuffer import ReplayBuffer
import EnvironmentUtility
from PolicyUtility import takeEpisodeStep


N_iterations = int(5e6)
targetUpdateFrequency = 10000
gamma = 0.99

updateFrequency = 4
frameHistoryLength = 4
minibatchSize = 32
bufferSize = 1000000
prefillSize = 50000

stepSize = 1e-4
beta1 = 0.9
beta2 = 0.999
adam_eps = 1e-4
video_log_frequency = 1000
adam_params=dict(N_iterations=N_iterations,
                step_size=stepSize,
                b1=beta1,
                b2=beta2,
                eps=adam_eps,
                )

learningParamsDict = {
    'params_v0':{
        'seed': 0,
        'eps_steps': [0, 1e6, 2.5e6],
        'eps_values': [0.2, 0.1, 0.01]
     },
    'params_v1':{
        'seed': 304,
        'eps_steps': [0, 1e6, 2.5e6],
        'eps_values': [0.2, 0.1, 0.01]
     },
    'params_v2':{
        'seed': 809,
        'eps_steps': [0, 1e6, 2.5e6],
        'eps_values': [0.2, 0.1, 0.01]
     },
    'params_v3':{
        'seed': 0,
        'eps_steps': [0, 1e6, 2.5e6],
        'eps_values': [0.5, 0.1, 0.001]
     },
    'params_v4':{
        'seed': 304,
        'eps_steps': [0, 1e6, 2.5e6],
        'eps_values': [0.5, 0.1, 0.001]
     },
    'params_v5':{
        'seed': 809,
        'eps_steps': [0, 1e6, 2.5e6],
        'eps_values': [0.5, 0.1, 0.001] 
     },
}

dim_nums=('NHWC', 'HWIO', 'NHWC')

def constructDuelNetwork(n_actions, seed, input_shape):
    advantage_stream = stax.serial(Dense(512), Relu, Dense(n_actions))

    state_function_stream = stax.serial(Dense(512), Relu, Dense(1))
    dueling_architecture = stax.serial(
                                    elementwise(lambda x: x/255.0),
                                    GeneralConv(dim_nums, 32, (8,8), strides=(4,4) ), 
                                    Relu,
                                    GeneralConv(dim_nums, 64, (4,4), strides=(2,2) ), 
                                    Relu,
                                    GeneralConv(dim_nums, 64, (3,3), strides=(1,1) ), 
                                    Relu,
                                    Flatten,
                                    FanOut(2),
                                    parallel(advantage_stream, state_function_stream),
                                )

    def duelingNetworkMapping(inputs):
        advantage_values = inputs[0]
        state_values = inputs[1]
        advantage_sums = jnp.sum(advantage_values, axis=1)

        advantage_sums = advantage_sums / float(n_actions)
        advantage_sums = advantage_sums.reshape(-1, 1)

        Q_values = state_values + (advantage_values - advantage_sums)
  
        return Q_values

    duelArchitectureMapping = jit(duelingNetworkMapping)

    ##### Create deep neural net
    model = DDQN(n_actions,
                 input_shape,
                 adam_params,
                 architecture=dueling_architecture,
                 seed=seed,
                 mappingFunction=duelArchitectureMapping)

    return model


def constructSingleStreamNetwork(n_actions, seed, input_shape):
    single_stream_architecture = stax.serial(
                                elementwise(lambda x: x/255.0),  # normalize
                                ### convolutional NN (CNN)
                                GeneralConv(dim_nums, 32, (8,8), strides=(4,4) ), 
                                Relu,
                                GeneralConv(dim_nums, 64, (4,4), strides=(2,2) ), 
                                Relu,
                                GeneralConv(dim_nums, 64, (3,3), strides=(1,1) ), 
                                Relu,
                                Flatten, # flatten output
                                Dense(1024), 
                                Relu,
                                Dense(n_actions)
                            )

    model = DDQN(n_actions,
                 input_shape,
                 adam_params,
                 architecture=single_stream_architecture,
                 seed=seed)

    return model;


def constructEnvironmentAndModel(seed, exportDir, isSingleStream=True):
    env = EnvironmentUtility.getEnvironment(seed, "Enduro-v0", video_log_frequency, exportDir)
    n_actions = env.action_space.n

    print("Is single stream {}.".format(isSingleStream))

    replayBuffer = ReplayBuffer(bufferSize, (env.observation_space.shape[0], env.observation_space.shape[1], 1), frameHistoryLength)

    frameShape = (env.observation_space.shape[0], env.observation_space.shape[1])
    inputShape = (1,) + frameShape + (frameHistoryLength,)

    model = constructSingleStreamNetwork(n_actions, seed, inputShape) if isSingleStream else constructDuelNetwork(n_actions, seed, inputShape)

    return (env, replayBuffer, model)



def fillBuffer(env, replayBuffer):
    totalTime = time.time()

    print("Start Prefilling buffer")
    prefill_index = 0

    while prefill_index < prefillSize:
        state = env.reset()
        isFinal = False

        while not isFinal:
            bufferIndex = replayBuffer.recordFrame(state)

            action = env.action_space.sample()
            nextFrame, reward, isFinal, _ = env.step(action)

            replayBuffer.recordEffect(prefill_index, action, reward, isFinal)
            
            state = nextFrame

            prefill_index += 1
    
    print("Finished prefilling the buffer")



def runLearningProcess(env, replayBuffer, model, paramsKey, saveDir):
    print("Start learning for {}\n".format(paramsKey))

    state = env.reset()
    params = learningParamsDict[paramsKey]
    epsilonSchedule = dict(epsilonThresholds=params['eps_steps'], epsilonValues=params['eps_values'])

    logger = Logger(env, epsilonSchedule)

    saveFilePath = saveDir + '/' + paramsKey
    for i in range(N_iterations):
        bufferIndex = replayBuffer.recordFrame(state)
        encodedLastObservation = replayBuffer.returnRecentObservation()
        encodedState = np.expand_dims(encodedLastObservation, 0)


        state, _, isTerminal = takeEpisodeStep(i, env, model,
                                        replayBuffer, bufferIndex, encodedState,
                                        epsilonSchedule=epsilonSchedule)

        if i % updateFrequency == 0:
            model.update(replayBuffer, minibatchSize, gamma)

        if i % targetUpdateFrequency == 0:
            model.updateTarget()

        if isTerminal:
            logger.stats(i, saveFilePath + '.json')

            state = env.reset()

    print("Final Result")

    logger.stats(5000000, saveFilePath + '.json')

    logger.plot(env.spec._env_name, plotFile=saveFilePath + '.png')


def runLearning(isSingleStream, keys, path):
    for key in keys:
        params = learningParamsDict[key]
        seed = params['seed']

        np.random.seed(seed)
        random.seed(seed)

        exportDir = os.path.join(path, key)
        if not os.path.exists(exportDir):
            os.mkdir(exportDir, 0o666)

        exportDir = os.path.join(path, key)

        print("Start learning for {}.".format(key))

        env, replayBuffer, model = constructEnvironmentAndModel(seed, exportDir, isSingleStream=isSingleStream)

        fillBuffer(env, replayBuffer)

        runLearningProcess(env, replayBuffer, model, key, exportDir)
    


