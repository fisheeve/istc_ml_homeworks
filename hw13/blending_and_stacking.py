import numpy as np
from sklearn.metrics import accuracy_score


def proba_to_num(proba):
    p = []
    for x in proba:
        if x > 0.5: p.append(1)
        else: p.append(0)
    return np.array(p)


class Blender:
    def __init__(self, models, blender, blend_size=0.3):
        self.models = models
        self.blend_size = blend_size
        self.blender = blender
        
    def train(self, data, y):
        split_index = int(self.blend_size * len(data))
        
        # model training
        train_data = data[:-split_index]
        train_y = y[:-split_index]
        for index in range(len(self.models)):
            self.models[index].fit(train_data, train_y)
                           
        # predict on test
        test_data = data[split_index:]
        test_y = y[split_index:]
        predictions = []
        for model in self.models:
            predictions.append(model.predict_proba(test_data)[:, 1])
        predictions = np.vstack((np.array(p) for p in predictions)).T
        blending_data = np.hstack((predictions, test_data))
        
        # train blender
        self.blender.fit(blending_data, test_y)
    
    def predict(self, data):
        predictions = [model.predict_proba(data)[:, 1]
                      for model in self.models]
        predictions = np.vstack((np.array(p) for p in predictions)).T
        blending_data = np.hstack((predictions, data))
        return self.blender.predict_proba(blending_data)        

    
class Stacking:
    def __init__(self, blender, blender_model, models, blend_size=0.3, test_size=0.2, folds=5):
        self.blender = blender
        self.blender_model = blender_model
        self.models = models
        self.test_size = test_size
        self.blend_size = blend_size
        self.folds = folds
        self.blenders = []
        
    def train(self, data, y):
        test_split = int(self.test_size*len(data))
        blenders = []
        score = []
        #blender training
        for i in range(self.folds):
            mask = np.arange(len(data))
            np.random.shuffle(mask)
            shuffled_data = data[mask]
            shuffled_y = y[mask]
            train_data = shuffled_data[:-test_split]
            test_data = shuffled_data[-test_split:]
            train_y = shuffled_y[:-test_split]
            test_y = shuffled_y[-test_split:]
            blender = self.blender(self.models, self.blender_model, self.blend_size)
            blender.train(train_data, train_y)
            pred = proba_to_num(blender.predict(test_data)[:,1])
            score.append(accuracy_score(pred, test_y))
            blenders.append(blender)
        #choosing best blender
        self.stacker = blenders[np.argmax(score)]
        
    def predict(self, new_data):
        return self.stacker.predict(new_data)