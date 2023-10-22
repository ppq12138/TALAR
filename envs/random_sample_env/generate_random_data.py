import copy
import random, numpy as np
from algorithms.demonstration import statePairDemonstration


def generateData(point_num=3, sample_num=1000, magnitude=1.2, random_generation=False):
    buffer = statePairDemonstration(stateSize=point_num * 2, bufferSize=sample_num)

    if random_generation:
        for data_idx in range(sample_num):
            former_data = []
            later_data = []
            for point_idx in range(point_num):
                former_data.append([random.uniform(-1, 1), random.uniform(-1, 1)])
                later_data.append([random.uniform(-1, 1), random.uniform(-1, 1)])
            buffer.addData(np.hstack(former_data), np.hstack(later_data))

    else:

        former_data = np.random.uniform(0, 1, size=(sample_num, point_num * 2))
        later_data = copy.deepcopy(former_data)
        random_idx = np.random.choice(list(range(point_num)), sample_num)
        batch_idx = np.arange(sample_num)
        later_data[batch_idx, random_idx * 2] += np.random.uniform(-1, 1, size=sample_num)
        later_data[batch_idx, random_idx * 2 + 1] += np.random.uniform(-1, 1, size=sample_num)

        for i in range(sample_num):
            buffer.addData(former_data[i], later_data[i])
    
    return buffer




if __name__ == "__main__":
    generateData()