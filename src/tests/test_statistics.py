import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
    home_path = home_path.parent
  
#data_src_path = home_path/'src'/'data'
utils_path = home_path/'src'/'utils'
  
# add ../utils/ into the path
if utils_path.as_posix() not in sys.path:
    sys.path.insert(0, utils_path.as_posix())

from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, \
roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import unittest
#import ../src/utils/statistics
from statistics import Statistics


class TestStatistics(unittest.TestCase):
    def test_new_exp1(self,st):
        st.new_experiment("loremipsum", {"batch_size": 64, "lambda": 17})
        n = "{} ".format(len(st.list_experiment) - 1) + "loremipsum"
        self.assertEqual(st.current_experiment_name, n)
        self.assertEqual(st.list_experiment[-1], n)
        self.assertEqual(st.param[n], {"batch_size": 64, "lambda": 17})
        self.assertEqual(st.balanced_accuracy[n], [])
        self.assertEqual(st.roc_area[n], [])
        self.assertEqual(st.batch_count[n], 0)
        self.assertEqual(st.epoch_count[n], 0)
    
    def test_new_ep(self,st):
        past_epoch_count = st.epoch_count[st.current_experiment_name]
        st.new_epoch()
        self.assertEqual(st.epoch_count[st.current_experiment_name], past_epoch_count + 1)
        
    def test_new_add_b1(self, st):
        pred = [0,0,0,0,1]
        real = [1,0,0,0,1]
        past_number_batch = st.batch_count[st.current_experiment_name]
        past_y_pred = list(st.y_pred)
        past_y_real = list(st.y_true)
        st.add_batch_results(pred,real)
        self.assertGreater(st.epoch_count[st.current_experiment_name], 0)
        if st.epoch_count[st.current_experiment_name] == 1:
            self.assertEqual(st.batch_count[st.current_experiment_name], past_number_batch + 1)
        else:
            self.assertEqual(st.batch_count[st.current_experiment_name], past_number_batch)
        self.assertEqual(st.y_pred, past_y_pred + pred)
        self.assertEqual(st.y_true, past_y_real + real)
        #print(len(st.balanced_accuracy[st.current_experiment_name]), st.epoch_count[st.current_experiment_name], st.batch_count[st.current_experiment_name])
        #assert len(st.balanced_accuracy[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
        #assert len(st.roc_area[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
        self.assertEqual(st.balanced_accuracy[st.current_experiment_name][-1], 0.875)
        self.assertEqual(st.roc_area[st.current_experiment_name][-1], 0.875)
        
    def test_new_exp2(self,st):
        st.new_experiment("alive", {"batch_size": 64, "beta": 17})
        n = "{} ".format(len(st.list_experiment) - 1) + "alive"
        self.assertEqual(st.current_experiment_name, n)
        self.assertEqual(st.list_experiment[-1], n)
        self.assertEqual(st.param[n], {"batch_size": 64, "lambda": 17})
        self.assertEqual(st.balanced_accuracy[n], [])
        self.assertEqual(st.roc_area[n], [])
        self.assertEqual(st.batch_count[n], 0)
        self.assertEqual(st.epoch_count[n], 0)
        
    def test_new_add_b2(self, st):
        pred = [0,0,0,0,1,0,1,1,1]
        real = [1,0,0,0,1,0,1,0,1]
        past_number_batch = st.batch_count[st.current_experiment_name]
        past_y_pred = list(st.y_pred)
        past_y_real = list(st.y_true)
        st.add_batch_results(pred,real)
        self.assertGreater(st.epoch_count[st.current_experiment_name], 0)
        if st.epoch_count[st.current_experiment_name] == 1:
            self.assertEqual(st.batch_count[st.current_experiment_name], past_number_batch + 1)
        else:
            self.assertEqual(st.batch_count[st.current_experiment_name], past_number_batch)
        self.assertEqual(st.y_pred, past_y_pred + pred)
        self.assertEqual(st.y_true, past_y_real + real)
        #print(len(st.balanced_accuracy[st.current_experiment_name]), st.epoch_count[st.current_experiment_name], st.batch_count[st.current_experiment_name])
        #assert len(st.balanced_accuracy[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
        #assert len(st.roc_area[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
        self.assertEqual(st.balanced_accuracy[st.current_experiment_name][-1], 0.775)
        self.assertEqual(st.roc_area[st.current_experiment_name][-1], 0.7750000000000001)

        
def new_exp1(st):
    st.new_experiment("loremipsum", {"batch_size": 64, "lambda": 17})
    n = "{} ".format(len(st.list_experiment) - 1) + "loremipsum"
    assert st.current_experiment_name == n
    assert st.list_experiment[-1] == n
    assert st.param[n] == {"batch_size": 64, "lambda": 17}
    assert st.balanced_accuracy[n] == []
    assert st.roc_area[n] == []
    assert st.batch_count[n] == 0
    assert st.epoch_count[n] == 0
    
def new_ep(st):
    past_epoch_count = st.epoch_count[st.current_experiment_name]
    st.new_epoch()
    assert st.epoch_count[st.current_experiment_name] == past_epoch_count + 1
        
def add_b1(st):
    pred = [0,0,0,0,1]
    real = [1,0,0,0,1]
    past_number_batch = st.batch_count[st.current_experiment_name]
    past_y_pred = list(st.y_pred)
    past_y_real = list(st.y_true)
    st.add_batch_results(pred,real)
    assert st.epoch_count[st.current_experiment_name] > 0
    if st.epoch_count[st.current_experiment_name] == 1:
        assert st.batch_count[st.current_experiment_name] == past_number_batch + 1
    else:
        assert st.batch_count[st.current_experiment_name] == past_number_batch
    assert st.y_pred == past_y_pred + pred
    assert st.y_true == past_y_real + real
    #print(len(st.balanced_accuracy[st.current_experiment_name]), st.epoch_count[st.current_experiment_name], st.batch_count[st.current_experiment_name])
    #assert len(st.balanced_accuracy[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
    #assert len(st.roc_area[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
    assert st.balanced_accuracy[st.current_experiment_name][-1] == 0.875
    assert st.roc_area[st.current_experiment_name][-1] == 0.875
    
def new_exp2(st):
    st.new_experiment("alive", {"batch_size": 64, "beta": 17})
    n = "{} ".format(len(st.list_experiment) - 1) + "alive"
    assert st.current_experiment_name == n
    assert st.list_experiment[-1] == n
    assert st.param[n] == {"batch_size": 64, "beta": 17}
    assert st.balanced_accuracy[n] == []
    assert st.roc_area[n] == []
    assert st.batch_count[n] == 0
    assert st.epoch_count[n] == 0

        
def add_b2(st):
    pred = [0,0,0,0,1,0,1,1,1]
    real = [1,0,0,0,1,0,1,0,1]
    past_number_batch = st.batch_count[st.current_experiment_name]
    past_y_pred = list(st.y_pred)
    past_y_real = list(st.y_true)
    st.add_batch_results(pred,real)
    assert st.epoch_count[st.current_experiment_name] > 0
    if st.epoch_count[st.current_experiment_name] == 1:
        assert st.batch_count[st.current_experiment_name] == past_number_batch + 1
    else:
        assert st.batch_count[st.current_experiment_name] == past_number_batch
    assert st.y_pred == past_y_pred + pred
    assert st.y_true == past_y_real + real
    #assert len(st.balanced_accuracy[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count + st.batch_count
    #assert len(st.roc_area[st.current_experiment_name]) == (st.epoch_count - 1) * st.batch_count + st.batch_count
    assert st.balanced_accuracy[st.current_experiment_name][-1] == 0.775
    assert st.roc_area[st.current_experiment_name][-1] == 0.7750000000000001
    
def main():
    stat = Statistics()
    new_exp1(stat)
    for e in range(3):
        new_ep(stat)
        for b in range(5):
            add_b1(stat)
    new_exp2(stat)
    for e in range(1):
        new_ep(stat)
        for b in range(10):
            add_b2(stat)

if __name__ == '__main__':
    #main()
    print("azda")
    unittest.main()
