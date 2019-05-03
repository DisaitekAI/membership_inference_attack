import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
    home_path = home_path.parent
  
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
import time
from statistics import Statistics


class TestStatistics(unittest.TestCase):

  def test_new_experiment(self):
    st = Statistics()
    st.new_experiment('loremipsum', {'batch_size': 64, 'lambda': 17})
    self.assertEqual(st.y_pred, [])
    self.assertEqual(st.y_true, [])
    self.assertEqual(st.exp, 
        [ { 'name': 'loremipsum', 'param': {'batch_size': 64, 'lambda': 17},
           'model_training': [], 'mia_stats' : { 'mia_train_in_distribution' : [],
                                                 'mia_train_out_distribution': [],
                                                 'mia_test_in_distribution'  : [],
                                                 'mia_test_out_distribution' : [] } } ] )
    self.assertIsNone(st.resume)

  def test_new_train(self):
    st = Statistics()
    st.new_experiment('loremipsum', {'batch_size': 64, 'lambda': 17})
    test_start = time.time()
    st.new_train('martel', 'hell')
    self.assertEqual(st.y_pred, [])
    self.assertEqual(st.y_true, [])
    self.assertEqual(len(st.exp), 1)
    self.assertEqual(st.exp[0]['name'], 'loremipsum')
    self.assertEqual(st.exp[0]['param'], {'batch_size': 64, 'lambda': 17})
    self.assertEqual(st.exp[0]['model_training'][0]['name'], 'martel')
    self.assertEqual(st.exp[0]['model_training'][0]['label'], 'hell')
    self.assertEqual(st.exp[0]['model_training'][0]['loss'], [])
    self.assertEqual(len(st.exp[0]['model_training'][0]['time']), 1)
    self.assertAlmostEqual(st.exp[0]['model_training'][0]['time'][0], test_start, places = 4)
    self.assertEqual(st.exp[0]['model_training'][0]['measures'], { 'balanced_accuracy': [], 
                                                                'roc_area'         : [], 
                                                                'report'           : '' } )
    self.assertEqual(st.exp[0]['mia_stats'], { 'mia_train_in_distribution' : [],
                                               'mia_train_out_distribution': [],
                                               'mia_test_in_distribution'  : [],
                                               'mia_test_out_distribution' : [] })

    def test_new_epoch(self):
      st = Statistics()
      st.new_experiment('loremipsum', {'batch_size': 64, 'lambda': 17})
      test_start = time.time()
      st.new_train('martel', 'hell')
      self.assertEqual(st.y_pred, [])
      self.assertEqual(st.y_true, [])
      self.assertEqual(len(st.exp), 1)
      self.assertEqual(st.exp[0]['name'], 'loremipsum')
      self.assertEqual(st.exp[0]['param'], {'batch_size': 64, 'lambda': 17})
      self.assertEqual(st.exp[0]['model_training'][0]['name'], 'martel')
      self.assertEqual(st.exp[0]['model_training'][0]['label'], 'hell')
      self.assertEqual(st.exp[0]['model_training'][0]['loss'], [ [] ])
      self.assertEqual(len(st.exp[0]['model_training'][0]['time']), 1)
      self.assertAlmostEqual(st.exp[0]['model_training'][0]['time'][0], test_start, places = 4)
      self.assertEqual(st.exp[0]['model_training'][0]['measures'], { 'balanced_accuracy': [], 
                                                                  'roc_area'         : [], 
                                                                  'report'           : '' } )
      self.assertEqual(st.exp[0]['mia_stats'], { 'mia_train_in_distribution' : [],
                                                 'mia_train_out_distribution': [],
                                                 'mia_test_in_distribution'  : [],
                                                 'mia_test_out_distribution' : [] } )
    def test_new_batch(self):
      st = Statistics()
      st.new_experiment('loremipsum', {'batch_size': 64, 'lambda': 17})
      test_start = time.time()
      st.new_train('martel', 'hell')
      pred = [0,0,0,0,1]
      real = [1,0,0,0,1]
      st.new_batch(pred,real)
      self.assertEqual(st.y_pred, pred)
      self.assertEqual(st.y_true, real)
      self.assertEqual(len(st.exp), 1)
      self.assertEqual(st.exp[0]['name'], 'loremipsum')
      self.assertEqual(st.exp[0]['param'], {'batch_size': 64, 'lambda': 17})
      self.assertEqual(st.exp[0]['model_training'][0]['name'], 'martel')
      self.assertEqual(st.exp[0]['model_training'][0]['label'], 'hell')
      self.assertEqual(st.exp[0]['model_training'][0]['loss'], [ [] ])
      self.assertEqual(len(st.exp[0]['model_training'][0]['time']), 1)
      self.assertAlmostEqual(st.exp[0]['model_training'][0]['time'][0], test_start, places = 4)
      self.assertEqual(st.exp[0]['model_training'][0]['measures'], { 'balanced_accuracy': [], 
                                                                  'roc_area'         : [], 
                                                                  'report'           : '' } )
      self.assertEqual(st.exp[0]['mia_stats'], { 'mia_train_in_distribution' : [],
                                                 'mia_train_out_distribution': [],
                                                 'mia_test_in_distribution'  : [],
                                                 'mia_test_out_distribution' : [] } )
        
#     def test_new_add_b1(self):
#         st = Statistics()
#         st.new_experiment('loremipsum', {'batch_size': 64, 'lambda': 17})
#         st.new_epoch()
#         pred = [0,0,0,0,1]
#         real = [1,0,0,0,1]
#         past_number_batch = st.batch_count[st.current_experiment_name]
#         past_y_pred = list(st.y_pred)
#         past_y_real = list(st.y_true)
#         st.add_batch_results(pred,real)
#         self.assertGreater(st.epoch_count[st.current_experiment_name], 0)
#         if st.epoch_count[st.current_experiment_name] == 1:
#             self.assertEqual(st.batch_count[st.current_experiment_name], past_number_batch + 1)
#         else:
#             self.assertEqual(st.batch_count[st.current_experiment_name], past_number_batch)
#         self.assertEqual(st.y_pred, past_y_pred + pred)
#         self.assertEqual(st.y_true, past_y_real + real)
#         #print(len(st.balanced_accuracy[st.current_experiment_name]), st.epoch_count[st.current_experiment_name], st.batch_count[st.current_experiment_name])
#         #assert len(st.balanced_accuracy[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
#         #assert len(st.roc_area[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
#         self.assertEqual(st.balanced_accuracy[st.current_experiment_name][-1], 0.875)
#         self.assertEqual(st.roc_area[st.current_experiment_name][-1], 0.875)
        
#     def test_new_exp2(self):
#         st = Statistics()
#         st.new_experiment('alive', {'batch_size': 64, 'beta': 17})
#         n = '{} '.format(len(st.list_experiment) - 1) + 'alive'
#         self.assertEqual(st.current_experiment_name, n)
#         self.assertEqual(st.list_experiment[-1], n)
#         self.assertEqual(st.param[n], {'batch_size': 64, 'beta': 17})
#         self.assertEqual(st.balanced_accuracy[n], [])
#         self.assertEqual(st.roc_area[n], [])
#         self.assertEqual(st.batch_count[n], 0)
#         self.assertEqual(st.epoch_count[n], 0)
        
#     def test_new_add_b2(self):
#         st = Statistics()
#         st.new_experiment('alive', {'batch_size': 64, 'beta': 17})
#         st.new_epoch()
#         pred = [0,0,0,0,1,0,1,1,1]
#         real = [1,0,0,0,1,0,1,0,1]
#         past_number_batch = st.batch_count[st.current_experiment_name]
#         past_y_pred = list(st.y_pred)
#         past_y_real = list(st.y_true)
#         st.add_batch_results(pred,real)
#         self.assertGreater(st.epoch_count[st.current_experiment_name], 0)
#         if st.epoch_count[st.current_experiment_name] == 1:
#             self.assertEqual(st.batch_count[st.current_experiment_name], past_number_batch + 1)
#         else:
#             self.assertEqual(st.batch_count[st.current_experiment_name], past_number_batch)
#         self.assertEqual(st.y_pred, past_y_pred + pred)
#         self.assertEqual(st.y_true, past_y_real + real)
#         #print(len(st.balanced_accuracy[st.current_experiment_name]), st.epoch_count[st.current_experiment_name], st.batch_count[st.current_experiment_name])
#         #assert len(st.balanced_accuracy[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
#         #assert len(st.roc_area[st.current_experiment_name]) == (st.epoch_count[st.current_experiment_name] - 1) * st.batch_count[st.current_experiment_name] + st.batch_count[st.current_experiment_name]
#         self.assertEqual(st.balanced_accuracy[st.current_experiment_name][-1], 0.775)
#         self.assertEqual(st.roc_area[st.current_experiment_name][-1], 0.7750000000000001)
    
# def main():
#     stat = Statistics()
#     new_exp1(stat)
#     for e in range(3):
#         new_ep(stat)
#         for b in range(5):
#             add_b1(stat)
#     new_exp2(stat)
#     for e in range(1):
#         new_ep(stat)
#         for b in range(10):
#             add_b2(stat)

if __name__ == '__main__':
    unittest.main()
