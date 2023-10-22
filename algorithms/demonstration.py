import random, os
import numpy as np

class BaseDemonstration():
    def __init__(self, stateSize, bufferSize=10000):
        self.bufferSize = bufferSize
        self.stateSize = stateSize
        self.ptr = 0
        self.sampleSize = 0
    
    def addData(self, *args):
        raise NotImplementedError
    
    def sampleBatchData(self, *args):
        pass
    
    def sampleOneData(self, *args):
        pass



class TrajectoryDemonstration(BaseDemonstration):

    def __init__(self, obsSize=10, stateSize=40, actionSize=1, goalSize=34, bufferSize=100000, max_traj_len=50):
        super(TrajectoryDemonstration, self).__init__(stateSize, bufferSize)
        self.actionSize = actionSize
        self.obsSize = obsSize
        self.goalSize = goalSize
        self.max_traj_len = max_traj_len

        self.fields = {}
        self.fields_attrs = {}
        self.fields_name = []
        fields = {
            'actions': {
                'shape': (self.max_traj_len, self.actionSize),
                'dtype': 'float32'
            },
            'rewards': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'float32'
            },
            'terminals': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'bool'
            },
            'valid': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'float32'
            },
            'observations': {
                'shape': (self.max_traj_len, self.obsSize),
                'dtype': 'float32'
            },
            'next_observations': {
                'shape': (self.max_traj_len, self.obsSize),
                'dtype': 'float32'
            },
            "states": {
                'shape': (self.max_traj_len, self.stateSize),
                'dtype': 'float32'
            },
            "goals": {
                'shape': (self.max_traj_len, self.goalSize),
                'dtype': 'float32'
            }
        }
        self.add_fields(fields)
    

    def add_fields(self, fields_attrs):
        self.fields_attrs.update(fields_attrs)

        for field_name, field_attrs in fields_attrs.items():
            field_shape = (self.bufferSize, *field_attrs['shape'])
            initializer = field_attrs.get('initializer', np.zeros)
            self.fields[field_name] = initializer(
                field_shape, dtype=field_attrs['dtype'])
        
        self.field_names = list(fields_attrs.keys())


    def addData(self, obs, action, next_action):
        pass
    
    
    def addTrajectoryData(self, trajectoryData: dict):
        for k in trajectoryData:
            self.fields[k][self.ptr][:len(trajectoryData[k])] = trajectoryData[k]

        self.sampleSize = (self.ptr + 1) if self.sampleSize < self.bufferSize else self.bufferSize
        self.ptr = (self.ptr + 1) % self.bufferSize


    def save(self, path):
        # if not os.path.exists(path):
        #     os.makedirs(path)
        
        data = {
            "fields": self.fields,
            "ptr":self.ptr,
            "sampleSize":self.sampleSize,
        }
        np.save(path, data)
    

    def save_partial(self, path, num_save):
        fields = {}
        total_idx = np.arange(self.sampleSize)
        np.random.shuffle(total_idx)
        indices = total_idx[:num_save]
        for k in self.fields.keys():
            if k in ["rewards", "actions", "terminals", "states", "goals"]:
                continue
            fields[k] = self.fields[k][indices]
        
        data = {
            "fields": fields,
            "ptr":num_save - 1,
            "sampleSize":num_save,
        }
        np.save(path, data)
    

    def load(self, path):
        if path[-3:] == "npy":
            data = np.load(path, allow_pickle=True).item()
        else:
            data = np.load(path, allow_pickle=True)
            k = data.files[0]
            data = data[k].item

        for k in data["fields"].keys():
            self.add_fields({k:{
                "shape": data["fields"][k].shape[-2:] if len(data["fields"][k].shape) > 2 else (1, ),
                "dtype": data["fields"][k].dtype
            }})
        
        self.fields = data["fields"]
        self.field_names = list(data["fields"].keys())
        
        self.ptr = data["ptr"]
        self.sampleSize = data["sampleSize"]
        self.bufferSize = self.sampleSize

        
    def sampleTrajectoryBatchData(self, batch_size, length, return_middel_sample=False):
        if isinstance(length, list):
            max_length = max(length)
        elif isinstance(length, int):
            max_length = length
        else:
            assert NotImplementedError, "unsupported length type"

        valids = np.sum(self.fields['valid'][:self.sampleSize, :-max_length+1], axis=1).squeeze(-1)
        first_ind = np.random.choice(np.arange(self.sampleSize), p=valids/np.sum(valids), size=(batch_size, ))
        second_ind = []
        lens = [length] * batch_size
        if not return_middel_sample:
            for ind, item in enumerate(first_ind):
                second_ind.append(np.random.randint(valids[item]))
        else:
            lens = []
            valids = np.sum(self.fields['valid'], axis=1).squeeze(-1)
            for ind, item in enumerate(first_ind):
                start = np.random.randint(valids[item])
                second_ind.append(start)
                lens.append(min(length, int(valids[item]) - start))
   
        indices = [(a, b, c) for a, b, c in zip(first_ind, second_ind, lens)]
        return self.batch_by_double_index(indices)
    
    
    def batch_by_double_index(self, indices):
        batch = {}
        for field in self.field_names:
            shapes = self.fields_attrs[field]["shape"]
            shapes = (len(indices), *shapes)
            if field == "description":
                shapes = (len(indices), *(1,))
            data = np.zeros(shapes, dtype=self.fields_attrs[field]["dtype"])
            if field == "description":
                for ind, item in enumerate(indices):
                    idx = np.random.choice(self.fields[field][item[0]].shape[0])
                    data[ind] = np.array(self.fields[field][item[0]][idx])
            else:
                for ind, item in enumerate(indices):
                    data[ind][:item[2]] = self.fields[field][item[0], item[1]:item[1]+item[2]]
            batch[field] = data
        return batch


    def pop_field(self, k):
        _ = self.fields.pop(k)
        _ = self.fields_attrs.pop(k)
        

class PreTrainDemonstration(TrajectoryDemonstration):
    def __init__(self, obsSize=10, stateSize=40, actionSize=1, bufferSize=100000, max_traj_len=50):
        super(TrajectoryDemonstration, self).__init__(stateSize, bufferSize)
        self.actionSize = actionSize
        self.obsSize = obsSize
        self.max_traj_len = max_traj_len

        self.fields = {}
        self.fields_attrs = {}
        self.fields_name = []
        fields = {
            'actions': {
                'shape': (self.max_traj_len, self.actionSize),
                'dtype': 'float32'
            },
            'rewards': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'float32'
            },
            'terminals': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'bool'
            },
            'valid': {
                'shape': (self.max_traj_len, *(1, )),
                'dtype': 'float32'
            },
            'observations': {
                'shape': (self.max_traj_len, self.obsSize),
                'dtype': 'float32'
            },
            'next_observations': {
                'shape': (self.max_traj_len, self.obsSize),
                'dtype': 'float32'
            },
            "states": {
                'shape': (self.max_traj_len, self.stateSize),
                'dtype': 'float32'
            },
            "goal_abs": {
                'shape': (self.max_traj_len, 3),
                'dtype': 'float32'
            },
        }
        self.add_fields(fields)
        