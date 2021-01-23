import random
import numpy as np




class ReplayBuffer(object):
    """description of class"""
    def __init__(self, size, frameShape, frameLength):
        self.frameLength = frameLength
        self.size = size
        self.nextIndex = 0
        self.recordedObservationsNum = 0

        self.observations = None
        self.actions = None
        self.rewards = None
        self.isFinalStateRecord = None

    def canSample(self, batchSize):
        return batchSize + 1 <= self.recordedObservationsNum

    def sampleUniqueIndeces(self, n):
        uniqueResults = []
        while len(uniqueResults) < n:
            candidate = random.randint(0, self.recordedObservationsNum - 2)
            if candidate not in uniqueResults:
                uniqueResults.append(candidate)
        return uniqueResults

    def returnSample(self, indeces):
        observationBatch = np.concatenate([self._encodeObservation(index)[None] for index in indeces], 0)
        actionsBatch = self.actions[indeces]
        rewardsBatch = self.rewards[indeces]
        nextObservationsBatch = np.concatenate([self._encodeObservation(index + 1)[None] for index in indeces], 0)
        isFinalStateBatch = np.array([1.0 if self.isFinalStateRecord[index] else 0.0 for index in indeces], dtype = np.float32)

        return observationBatch, actionsBatch, rewardsBatch, nextObservationsBatch, isFinalStateBatch

    def recordFrame(self, frame):
        if self.observations is None:
            self.observations = np.empty(
                [self.size] + list(frame.shape),
                dtype=np.uint8)
            
            self.actions = np.empty([self.size], dtype=np.int32)
            self.rewards = np.empty([self.size], dtype=np.float32)
            self.isFinalStateRecord = np.empty([self.size], dtype=np.bool)

        self.observations[self.nextIndex] = frame
        
        returnIndex = self.nextIndex
        self.nextIndex = (self.nextIndex + 1) % self.size
        self.recordedObservationsNum = min(self.size, self.recordedObservationsNum + 1)

        return returnIndex

    def recordEffect(self, index, action, reward, done):
        self.actions[index] = action
        self.rewards[index] = reward
        self.isFinalStateRecord[index] = done

    def sample(self, batchSize):
        assert self.canSample(batchSize)
        indeces = self.sampleUniqueIndeces(batchSize)

        return self.returnSample(indeces)

    def returnRecentObservation(self):
        assert self.recordedObservationsNum > 0
        return self._encodeObservation((self.nextIndex - 1) % self.size)

    def _encodeObservation(self, index):
        sliceEnd = index + 1
        sliceStart = sliceEnd - self.frameLength

        if len(self.observations.shape) == 2:
            return self.observations[sliceEnd - 1]

        if sliceStart < 0 and self.recordedObservationsNum < self.size:
            sliceStart = 0

        for i in range(sliceStart, sliceEnd - 1):
            if self.isFinalStateRecord[i % self.size]:
                sliceStart = i + 1

        missingFramesLength = self.frameLength - (sliceEnd - sliceStart)

        if sliceStart < 0 or missingFramesLength > 0:
            frames = [np.zeros_like(self.observations[0]) for _ in range(missingFramesLength)]
            for index in range(sliceStart, sliceEnd):
                frames.append(self.observations[index % self.size])

            return np.concatenate(frames, 2)

        else:
            height, width = self.observations.shape[1], self.observations.shape[2]

            return (self.observations[sliceStart:sliceEnd].transpose(1, 2, 0, 3).reshape(height, width, -1))

   

