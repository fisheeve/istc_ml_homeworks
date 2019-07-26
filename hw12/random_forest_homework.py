
import numpy as np

class DecisionNode:
    def __init__(self, data, y, right_node = None, left_node = None, 
                 f_index = None, f_value = None, is_leaf=False):
        self.data = data
        self.y = y
        self.right_node = right_node
        self.left_node = left_node
        self.f_index = f_index
        self.f_value = f_value
        self.is_leaf = is_leaf


class DecisionTree:
    def __init__(self, loss="entropy", max_depth=5):
        if loss == "entropy":
            self.loss = self.entropy
        if loss == "gini":
            self.loss = self.gini
        if loss == "reg_loss":
            self.loss = self.reg_loss
        self.max_depth=max_depth
        
    def entropy(self, targets):
        entropy = 0
        classes = np.unique(targets)
        for item in classes:
            prob = len(targets[targets == item])/len(targets)
            entropy += -prob*np.log(prob)
        return entropy
            
    def gini(self,  targets):
        entropy = 0
        for item in classes:
            prob = len(targets[targets == item])/len(targets)
            entropy += prob*(1-prob)
        return entropy
    
    def reg_loss(self, targets):
        mean = targets.mean()
        return np.mean((targets-mean)**2)
    
    def train(self, data, y):
        self.tree = self.iterate(data, y)
        
    def iterate(self, data, y, current_depth=0):
        if len(y) == 1:
            return DecisionNode(data, y, is_leaf = True)
        h = self.loss(y)
        best_f, best_value = None, None
        for index, f in enumerate(data.T):
            for f_value in np.arange(min(f), max(f), 
                                    (max(f) - min(f)) / 50):
                h1 = self.loss(y[f >= f_value])
                h2 = self.loss(y[f < f_value])
                if h1 + h2 < h:
                    h = h1 + h2
                    best_f, best_value = index, f_value
                    data1 = data[f >= f_value] 
                    data2 = data[f < f_value]
                    y1 = y[f >= f_value]
                    y2 = y[f < f_value]
                
        if best_f is None or current_depth == self.max_depth:
            return DecisionNode(data, y, is_leaf = True)
        else:
            return DecisionNode(data, y, 
                right_node = self.iterate(data1, y1, current_depth + 1),
                left_node = self.iterate(data2, y2, current_depth + 1), 
                f_index = best_f, f_value = best_value, 
                is_leaf = False)
    
    def predict(self, point):
        node = self.tree
        
        while True:
            if node.is_leaf:
                counts = np.bincount(node.y)  
                return np.argmax(counts)
                    
            if point[node.f_index] >= node.f_value:
                node = node.right_node
            else:
                node = node.left_node

                

class RandomForest:
    def __init__(self, loss = 'entropy', max_depth = 3, 
                 num_tree = 2, data_part = 0.5):
        self.num_tree = num_tree
        self.data_part = data_part
        self.loss = loss
        self.max_depth = max_depth
        self.forest = []
        
    def train(self, data, targets):
        data_len = len(targets)
        mask = np.array(range(data_len))
        for i in range(self.num_tree):
            np.random.shuffle(mask)
            num_datum = int(self.data_part*data_len)
            dataset_mask = mask[:num_datum]
            dataset = data[dataset_mask]
            targetset = targets[dataset_mask]
            tree = DecisionTree(loss=self.loss, 
                               max_depth=self.max_depth)
            tree.train(dataset, targetset)
            self.forest.append(tree)
        
    def predict(self, point):
        prediction = []
        for tree in self.forest:
            prediction.append(tree.predict(point))
        counts = np.bincount(prediction)
        return np.argmax(counts)           
    