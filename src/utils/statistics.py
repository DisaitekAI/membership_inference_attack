from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pdb
import os, pathlib, sys
import torch
import time
import pdb
from collections import defaultdict

class Statistics:
  """class used to records statistical data for all experiments.
  """
  
  def __init__(self):
    self.y_pred    = []
    self.y_true    = []
    self.exp       = []
    self.resume    = None

  def new_experiment(self, name, parameters):
    """declares that a new experiment will be executed.
    
    Args:
      name (string): name of the experiment
      
      parameters (Dict): all parameters of the experiment
      
    """
    
    self.process_batchs()
    self.close_timer()
    experiment = { 'name': name, 'param': parameters, 'model_training': [], 'mia_stats' : { 'mia_train_in_distribution' : [],
                                                                                            'mia_train_out_distribution': [],
                                                                                            'mia_test_in_distribution'  : [],
                                                                                            'mia_test_out_distribution' : [] } }
    
    self.exp.append(experiment)

  def new_train(self, name = None, label = None):
    """declares that the training of a new model will be executed.
    
    Args:
      name (string): name of the model. If None the results of the model testing will not be printed individualy.
    
      label (string): label of the group to which belongs the model. Special processing are done with models of the same group (averaged accuracy for instance). 
    """
    self.process_batchs()
    self.close_timer()
    start = time.time()
    model = { 'name': name, 'label': label, 'measures': {'balanced_accuracy': [], 'roc_area': [], 'report': ''}, 'loss': [], 'time': [start] }
    self.exp[-1]['model_training'].append(model)

  def new_epoch(self):
    """declares that a new epoch of a training cycle will be executed.
    """
    self.process_batchs()
    self.exp[-1]['model_training'][-1]['loss'].append([])
    
        
  def new_batch(self, batch_pred, batch_true):
    """collects the results of a train epoch on the test dataset.
    
    Args:
      batch_pred (list(label)): list of the labels predicted by the ML for the current batch.
      
      batch_true (list(label)): list of the true labels for the current batch.
    """
    self.y_pred.extend(batch_pred)
    self.y_true.extend(batch_true)

  def process_batchs(self):
    """process the data for all saved batches.
    """
    if len(self.y_true) != 0:
      accuracy = balanced_accuracy_score(self.y_pred, self.y_true)
      self.exp[-1]['model_training'][-1]['measures']['balanced_accuracy'].append(accuracy)
      try:
        area = roc_auc_score(self.y_pred, self.y_true)
        self.exp[-1]['model_training'][-1]['measures']['roc_area'].append(area)
      except:
        self.exp[-1]['model_training'][-1]['measures']['roc_area'] = None
      report = classification_report(self.y_pred, self.y_true)
      self.exp[-1]['model_training'][-1]['measures']['report'] = report
      self.y_true = []
      self.y_pred = []

  def save(self, dir):
    """save the results of all experiments in dir
    """
    print("\n\nRecording...")


    self.process_batchs()
    self.close_timer()

    actual_reports = [f for f in dir.iterdir() if "Statistics_report_" in f.name]
    path = dir/"Statistics_report_{}".format(len(actual_reports))
    
    resume_path = path/'resume'
    os.makedirs(os.path.dirname(str(resume_path)), exist_ok=True)
    #os.makedirs(resume_path, exist_ok=True)
    with open(resume_path, 'w') as resume_file:
      sys.stdout = resume_file
      self.print_results()
    resume_file.closed

    for experiment in self.exp:
      for model in experiment['model_training']:
        if model['name'] is not None:
          
          for measure_name, measure_values in model['measures'].items():

            if measure_values is None:
                continue

            if type(measure_values) == list:
              plot_path = path/experiment['name']/model['name']/measure_name
              os.makedirs(os.path.dirname(str(plot_path)), exist_ok=True)
              plt.plot(measure_values)
              plt.title(measure_name)
              plt.savefig(plot_path)

          loss_path = path/experiment['name']/model['name']/'loss_curve'
          os.makedirs(os.path.dirname(str(loss_path)), exist_ok=True)
          plt.plot(model['loss'][-1])
          plt.title('loss evolution during training')
          plt.savefig(loss_path)

    #print(" Done.")

  def create_resume(self):
    """create the results resume of all experiments as a string and save it, if it's not already did
    """

    if self.resume is None:
      self.resume = ""

      for experiment in self.exp:
        self.resume += "   Experiment " + experiment['name'] + " :\n"
        self.resume += "Parameters :\n"
        self.resume += str(experiment['param'])

      groups = defaultdict(list)

      for model in experiment['model_training']:
          
        if model['name'] is not None:
          self.resume += "\n   Model " + model['name'] + " :\n"
          for measure_name in model['measures']:
            if model['measures'][measure_name] is None:
                continue
            self.resume += "\n" + measure_name
            self.resume += str(model['measures'][measure_name])
          self.resume += f"\nTraining time: {model['time'][1] - model['time'][0]}"

        if model['label'] is not None:
          groups[model['label']].append(model)

      #for mia in experiment['mia_stats']:
       # if len(mia['mia_train_in_distribution']):
        #  class_number = len(mia['mia_train_in_distribution'])
          
         # self.resume += "\nMembership mean distributions:"
          #for i in range(class_number):
           # self.resume += f"  class {i}"
            #for label, tab in mia.items():
             # self.resume += f"    {label}: {tab[i]}"

      for group_label, group in groups.items():
        self.resume += "\n\nAverage statistics of the group " + group_label + ":"
        
        for measure_name in group[0]['measures']: # every models with same label must have same measures

          values = [model['measures'][measure_name] for model in group]

          if None in values:
            continue

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

          if average is not None:
            self.resume += "\n  * " + measure_name + " : "
            self.resume += str(average)

        durations = [model['time'][1] - model['time'][0] for model in group]
        self.resume += f"\n\nAverage training time for the group {group_label}: {sum(durations) / len(durations)}"

  def print_results(self):
    """print the results of all experiments
    """
    self.process_batchs()
    self.close_timer()
    self.create_resume()
    print(self.resume)
          
  def process_mia_dataset(self, dataset):
    """process the mean distribution of input samples from a MIA dataset
    
    Args:
      dataset (torch Dataset): MIA train or test dataset
      
    Returns
      (mean_in_sample (torch Tensor), mean_out_sample (torch Tensor))
      
    """
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
    """process the mean distribution of all train and test MIA datasets 
    """
    current_mia_stats = self.exp[-1]['mia_stats']
    
    for dataset in train_datasets:
      mean_in, mean_out = self.process_mia_dataset(dataset)
      current_mia_stats['mia_train_in_distribution'].append(mean_in)
      current_mia_stats['mia_train_out_distribution'].append(mean_out)
      
    for dataset in test_datasets:
      mean_in, mean_out = self.process_mia_dataset(dataset)
      current_mia_stats['mia_test_in_distribution'].append(mean_in)
      current_mia_stats['mia_test_out_distribution'].append(mean_out)

  def close_timer(self):
    """
    save the end time of the last model
    """

    end = time.time()
    last_model = None
    if self.exp != []:
      if self.exp[-1]['model_training'] == []:
        if len(self.exp) > 1:
          if self.exp[-2]['model_training'] != []:
            last_model = self.exp[-2]['model_training'][-1]
      else:
        last_model = self.exp[-1]['model_training'][-1]
    
    if last_model is not None and len(last_model['time']) == 1:
      last_model['time'].append(end)

  def add_loss(self, loss):
    """
    collect the loss on the last batch during the training on the training set
    """
    self.exp[-1]['model_training'][-1]['loss'][-1].append(loss)