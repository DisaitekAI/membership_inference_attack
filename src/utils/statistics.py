from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pdb
import os, pathlib, sys
import torch

class Statistics:
  def __init__(self):
    self.y_pred    = []
    self.y_true    = []
    self.exp       = []
    self.mia_stats = []

  def new_experiment(self, name, parameters):
    self.process_batchs()
    experiment = { 'name': name, 'param': parameters, 'model_training': [] }
    
    mia = { 'mia_train_in_distribution' : [],
            'mia_train_out_distribution': [],
            'mia_test_in_distribution'  : [],
            'mia_test_out_distribution' : [] }
    
    self.exp.append(experiment)
    self.mia_stats.append(mia)

  def new_train(self, name = None, label = None):
    """
    define a new model training.

    :name name of the training. 
    :label group model training with the same label. 
    """
    self.process_batchs()
    model = { 'name': name, 'label': label, 'measures': {'balanced_accuracy': [], 'roc_area': [], 'report': ''} }
    self.exp[-1]['model_training'].append(model)

  def new_epoch(self):
    self.process_batchs()
    
        
  def new_batch(self, batch_pred, batch_true):
    self.y_pred.extend(batch_pred)
    self.y_true.extend(batch_true)

  def process_batchs(self):
    if len(self.y_true) != 0:
      accuracy = balanced_accuracy_score(self.y_pred, self.y_true)
      self.exp[-1]['model_training'][-1]['measures']['balanced_accuracy'].append(accuracy)
      # ~ area = roc_auc_score(self.y_pred, self.y_true)
      # ~ self.exp[-1]['model_training'][-1]['measures']['roc_area'].append(area)
      report = classification_report(self.y_pred, self.y_true)
      self.exp[-1]['model_training'][-1]['measures']['report'] = report
      self.y_true = []
      self.y_pred = []

  def save(self, log_dir):
    self.process_batchs()

    actual_reports = [f for f in log_dir.iterdir() if "Statistics_report_" in f.name]
    path = log_dir/"Statistics_report_{}".format(len(actual_reports))
    
    log_path = path/'Logs'
    os.makedirs(os.path.dirname(str(log_path)), exist_ok=True)
    with open(str(log_path), 'w') as log_file:
      sys.stdout = log_file
      self.print_results()
    log_file.closed

    for experiment in self.exp:
      for model in experiment["model_training"]:
        if model["name"] != None:
          for measure_name, measure_values in model["measures"].items():
            if type(measure_values) == list:
              plot_path = path/experiment["name"]/model["name"]/measure_name
              os.makedirs(os.path.dirname(str(plot_path)), exist_ok=True)
              plt.plot(measure_values)
              plt.title(measure_name)
              plt.savefig(str(plot_path))

  def print_results(self):
    self.process_batchs()

    for experiment in self.exp:
      print("   Experiment " + experiment["name"] + " :\n")
      print("Parameters :\n")
      print(experiment["param"])

      groups = dict()

      for model in experiment["model_training"]:
          
        if model["name"] != None:
          print("\n   Model " + model["name"] + " :\n")
          for measure_name in model["measures"]:
            print("\n" + measure_name)
            print(model["measures"][measure_name])

        if model["label"] != None:
          if model["label"] in groups:
            groups[model["label"]].append(model)
          else:
            groups[model["label"]] = [model]
            
      for mia in self.mia_stats:
        if len(mia['mia_train_in_distribution']):
          class_number = len(mia['mia_train_in_distribution'])
          
          print('\nMembership mean distributions:')
          for i in range(class_number):
            print(f"  class {i}")
            for label, tab in mia.items():
              print(f"    {label}: {tab[i]}")

    for group_label, group in groups.items():
      print("\nAverage statistics of the group " + group_label + ":\n")
      for measure_name in group[0]["measures"]: # every models with same label must have same measures
        values = [model["measures"][measure_name] for model in group]
        average = None
   
        if type(values[0]) == float:
          average = sum(values)/len(values)
          
        if type(values[0]) == list:
          final_values = []
          for l in values:
            if len(l):
              final_values.append(l[-1])
          
          if len(final_values):
            average = sum(final_values) / len(final_values)

        if average != None:
          print("\n" + measure_name + " : ")
          print(average)
          
  def process_mia_dataset(self, dataset):
    # iterate through attack model classes
    s_in  = []
    s_out = []
    
    #iterate through the samples
    for s_input, s_output in dataset:
      if s_output == 1:
        s_in.append(s_input)
      else:
        s_out.append(s_input)
        
    s_in  = torch.exp(torch.stack(s_in))
    s_out = torch.exp(torch.stack(s_out))
    
    return s_in.mean(dim = 0), s_out.mean(dim = 0)
    
  def membership_distributions(self, train_datasets, test_datasets):
    current_mia_stats = self.mia_stats[-1]
    
    for dataset in train_datasets:
      mean_in, mean_out = self.process_mia_dataset(dataset)
      current_mia_stats['mia_train_in_distribution'].append(mean_in)
      current_mia_stats['mia_train_out_distribution'].append(mean_out)
      
    for dataset in test_datasets:
      mean_in, mean_out = self.process_mia_dataset(dataset)
      current_mia_stats['mia_test_in_distribution'].append(mean_in)
      current_mia_stats['mia_test_out_distribution'].append(mean_out)
      
