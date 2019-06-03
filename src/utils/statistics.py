import pathlib, sys

home_path = pathlib.Path('.').resolve()
while home_path.name != 'membership_inference_attack':
  home_path = home_path.parent
  
reports_path = home_path/'reports'

from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, \
                            classification_report, roc_curve
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pdb
import os, pathlib, sys
import torch
import time
import pdb
from collections import defaultdict
import seaborn as sns
import pandas as pd

class Statistics:
  """class used to records statistical data for all experiments.
  """
  
  def __init__(self):
    self.y_pred    = []
    self.y_true    = []
    self.exp       = []
    self.resume    = None

  def new_experiment(self, name, parameters, label = None):
    """declares that a new experiment will be executed.
    
    Args:
      name (string): name of the experiment
      
      parameters (Dict): all parameters of the experiment

      label (Dict): contains infos to aggregate with similar experiments
      
    """
    self._call_stat("new_experiment")
    self._process_batchs()
    self._close_timer()
    experiment = { 'name': name, 'label': label, 'param': parameters, 'model_training': [], 
                   'mia_stats' : { 'mia_train_in_distribution' : [],
                                   'mia_train_out_distribution': [],
                                   'mia_test_in_distribution'  : [],
                                   'mia_test_out_distribution' : [] }, 
                    'mean_accuracies': None }
    
    self.exp.append(experiment)

  def new_train(self, name = None, label = None):
    """declares that the training of a new model will be executed.
    
    Args:
      name (string): name of the model. If None the results of the model testing will not be printed individualy.
    
      label (string): label of the group to which belongs the model. Special processing are done with models of the same group (averaged accuracy for instance). 
    """
    self._call_stat("new_train")
    self._process_batchs()
    self._close_timer()
    start = time.time()
    model = { 'name': name, 'label': label, 'loss': [], 'time': [start], 'measures': { 'balanced_accuracy': [], 
                                                                                       'roc_area'         : [], 
                                                                                       'report'           : '' } }
    self.exp[-1]['model_training'].append(model)

  def new_epoch(self):
    """declares that a new epoch of a training cycle will be executed.
    """
    self._call_stat("new_epoch")
    self._process_batchs()
    
        
  def new_batch(self, batch_pred, batch_true):
    """collects the results of a train epoch on the test dataset.
    
    Args:
      batch_pred (list(label)): list of the labels predicted by the ML for the current batch.
      
      batch_true (list(label)): list of the true labels for the current batch.
    """
    # self._call_stat("new_batch")
    self.y_pred.extend(batch_pred)
    self.y_true.extend(batch_true)

  def _process_batchs(self):
    """process the data for all saved batches.
    """
    if len(self.y_true) != 0:
      accuracy = balanced_accuracy_score(self.y_true, self.y_pred)
      self.exp[-1]['model_training'][-1]['measures']['balanced_accuracy'].append(accuracy)
      try:
        # if 'MIA model' in self.exp[-1]['model_training'][-1]['name']:
        #     pdb.set_trace()
        area = roc_auc_score(self.y_true, self.y_pred)
        self.exp[-1]['model_training'][-1]['measures']['roc_area'].append(area)
      except TypeError:
        self.exp[-1]['model_training'][-1]['measures']['roc_area'] = None
      except ValueError:
        if self.exp[-1]['model_training'][-1]['measures']['roc_area'] is not None:
            self.exp[-1]['model_training'][-1]['measures']['roc_area'].append(0.5) #not satisfying !!
      report = classification_report(self.y_true, self.y_pred)
      self.exp[-1]['model_training'][-1]['measures']['report'] = report
      self.y_true = []
      self.y_pred = []

  def _final_process(self):
    for idx in range(len(self.exp)):
      if self.exp[idx]['mean_accuracies'] is None and self.exp[idx]['label'] is not None:
        groups_exp = defaultdict(list)

        for experiment_idx, experiment in enumerate(self.exp):
          if experiment['label'] is not None:
            groups_exp[experiment['label']].append((experiment_idx,experiment))

        for group_label, group in groups_exp.items():

          mean_accuracies = [ sum(mia_model_accuracies) / len(mia_model_accuracies) \
            for mia_model_accuracies in \
              [ [ model['measures']['balanced_accuracy'][-1] for model in experiment['model_training'] if model['name'] is not None and 'mia' in model['name'].lower() ] ] \
            for (_,experiment) in group ]

          average = sum(mean_accuracies) / len(mean_accuracies)

          for (experiment_idx,_) in group:
            self.exp[experiment_idx]['mean_accuracies'] = mean_accuracies     


  def save(self, dir):
    """save the results of all experiments in dir
    """
    print("\n\nRecording...", end = ' ')


    self._process_batchs()
    self._final_process()
    self._close_timer()

    basename_report = 'Statistics_report'
    actual_reports = [f for f in dir.iterdir() if basename_report in f.name]
    file = f"{basename_report}_{len(actual_reports)}"
    path = dir/file
    print(f"in {file}...", end = ' ')

    resume_path = path/'resume'
    os.makedirs(os.path.dirname(str(resume_path)), exist_ok=True)
    with open(resume_path, 'w') as resume_file:
      self._create_resume()
      resume_file.write(self.resume)
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
              plt.clf()

          loss_path = path/experiment['name']/model['name']/'loss_curve'
          os.makedirs(os.path.dirname(str(loss_path)), exist_ok=True)
          plt.plot(model['loss'])
          plt.title('loss evolution during training')
          plt.savefig(loss_path)
          plt.clf()

      if experiment['label'] is not None:
        mean_path = path/f"Mean_accuracies_curve of experiment '{experiment['label']}'"
        os.makedirs(os.path.dirname(str(mean_path)), exist_ok=True)
        plt.plot(experiment['label']['interest_parameter_range'], experiment['mean_accuracies'])
        plt.title('Mean attack model accuracy variation')
        plt.xlabel(f"Different {experiment['label']} values")
        plt.ylabel('Mean mia accuracy')
        plt.savefig(mean_path)
        plt.clf()


    print("Done.")

  def _create_resume(self):
    """create the results resume of all experiments as a string and save it, if it's not already did
    """

    if self.resume is None:
      lines = []

      groups_exp = defaultdict(list)

      for experiment_idx, experiment in enumerate(self.exp):
        lines.append(f"   Experiment {experiment['name']} :")
        lines.append("Parameters :")
        lines.append(str(experiment['param']))

        if experiment['label'] is not None:
          groups_exp[experiment['label']].append((experiment_idx,experiment))

        groups_mod = defaultdict(list)

        for model in experiment['model_training']:
          
          if model['name'] is not None:
            lines.append(f"\n   Model {model['name']} :\n")
            for measure_name in model['measures']:
              if model['measures'][measure_name] is None:
                  continue
              lines.append(f"\n{measure_name}")
              lines.append(str(model['measures'][measure_name]))
            lines.append(f"\nTraining time: {model['time'][1] - model['time'][0]:3.5f}s")

          if model['label'] is not None:
            groups_mod[model['label']].append(model)

        mia = experiment['mia_stats']
        if len(mia['mia_train_in_distribution']):
          class_number = len(mia['mia_train_in_distribution'])
      
          lines.append('\nMembership mean distributions:')
          for i in range(class_number):
            lines.append(f"  class {i}")
            for label, tab in mia.items():
              lines.append(f"    {label}: {tab[i]}")

        for group_label, group in groups_mod.items():
          lines.append(f"\n\nAverage statistics of the group of model '{group_label}':")
        
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
              lines.append(f"\n  * {measure_name}: {average}")

          durations = [model['time'][1] - model['time'][0] for model in group]
          lines.append(f"\n\nAverage training time for the group {group_label}: {sum(durations) / len(durations):3.5f}s")

      for group_label, group in groups_exp.items():
        lines.append(f"\n\nAverage performances of the group of experiments '{group_label}':")

        mean_accuracies = [ sum(mia_model_accuracies) / len(mia_model_accuracies) \
          for mia_model_accuracies in \
            [ [ model['measures']['balanced_accuracy'][-1] for model in experiment['model_training'] if model['name'] is not None and 'mia' in model['name'].lower() ] ] \
          for (_,experiment) in group ]

        average = sum(mean_accuracies) / len(mean_accuracies)
        lines.append(f"Average mean accuracy of the group of experiments '{group_label}': {average}")

        for (experiment_idx,_) in group:
          self.exp[experiment_idx]['mean_accuracies'] = mean_accuracies

      self.resume = '\n'.join(lines)

  def print_results(self):
    """print the results of all experiments
    """
    self._process_batchs()
    self._close_timer()
    self._create_resume()
    print(self.resume)
          
  def _process_mia_dataset(self, dataset, klass, save_path):
    """process the mean distribution of input samples from a MIA dataset
    
    Args:
      dataset (torch Dataset): MIA train or test dataset
      
    Returns
      (mean_in_sample (torch Tensor), mean_out_sample (torch Tensor))
      
    """
    s_in  = []
    s_out = []
    s_in_miss  = []
    s_out_miss = []
    
    for s_input, s_output in dataset:
      if torch.argmax(s_input).item() == klass:
        if s_output == 1:
          s_in.append(s_input)
        else:
          s_out.append(s_input)
      else:
        if s_output == 1:
          s_in_miss.append(s_input)
        else:
          s_out_miss.append(s_input)
    
    s_in       = torch.exp(torch.stack(s_in))
    s_out      = torch.exp(torch.stack(s_out))
    
    if len(s_in_miss) != 0 and len(s_out_miss) != 0:
      s_in_miss  = torch.exp(torch.stack(s_in_miss))
      s_out_miss = torch.exp(torch.stack(s_out_miss))
    else:
      s_in_miss  = None
      s_out_miss = None
    
    df_in       = pd.DataFrame(s_in.numpy())
    df_out      = pd.DataFrame(s_out.numpy())
    
    df_in_miss  = None
    df_out_miss = None
    if s_in_miss is not None:
      df_in_miss  = pd.DataFrame(s_in_miss.numpy())
      df_out_miss = pd.DataFrame(s_out_miss.numpy())
    
    df_in['training set']  = ['in'  for i in range(len(df_in[0]))]
    df_out['training set'] = ['out' for i in range(len(df_out[0]))]
    
    if s_in_miss is not None:
      df_in_miss['training set']  = ['in'  for i in range(len(df_in_miss[0]))]
      df_out_miss['training set'] = ['out' for i in range(len(df_out_miss[0]))]
    
    in_count  = len(df_in)
    out_count = len(df_out)
    type_str_in  = f"true positive\nin: {in_count}"
    type_str_out = f"true positive\nout: {out_count}"
    df_in['type']  = [type_str_in for i in range(len(df_in[0]))]
    df_out['type'] = [type_str_out for i in range(len(df_out[0]))]
    
    if s_in_miss is not None:
      in_count  = len(df_in_miss)
      out_count = len(df_out_miss)
      type_str_in  = f"false negative\nin: {in_count}"
      type_str_out = f"false negative\nout: {out_count}"
      df_in_miss['type']  = [type_str_in for i in range(len(df_in_miss[0]))]
      df_out_miss['type'] = [type_str_out for i in range(len(df_out_miss[0]))]
    
    df = df_in.append(df_out, ignore_index = True)
    target_column = f"confidence for class {klass}"
    df_plot = df.rename(columns = { klass : target_column }) 

    sns.violinplot(x = 'type', y = target_column, 
                   data = df_plot[[target_column, 'training set', 'type']], 
                   scale = 'count', bw = 0.1, cut = 0)
    
    plt.savefig(save_path.as_posix() + "_tp.pdf")
    plt.clf()
    
    if s_in_miss is not None:
      df = df_in_miss.append(df_out_miss, ignore_index = True)
      target_column = f"confidence for class {klass}"
      df_plot = df.rename(columns = { klass : target_column }) 

      sns.violinplot(x = 'type', y = target_column, 
                     data = df_plot[[target_column, 'training set', 'type']], 
                     scale = 'count', bw = 0.1, cut = 0)
      
      plt.savefig(save_path.as_posix() + "_fn.pdf")
      plt.clf()
        
      # adding all samples under the label "all" to view the whole distribution
      in_count  = len(df_in) + len(df_in_miss)
      out_count = len(df_out) + len(df_out_miss)
      type_str_in  = f"all\nin: {in_count}"
      type_str_out = f"all\nout: {out_count}"
      df_in['type']       = [type_str_in  for i in range(len(df_in[0]))]
      df_out['type']      = [type_str_out for i in range(len(df_out[0]))]
      df_in_miss['type']  = [type_str_in  for i in range(len(df_in_miss[0]))]
      df_out_miss['type'] = [type_str_out for i in range(len(df_out_miss[0]))]
      
      df = df_in.append(df_out, ignore_index = True)
      df = df.append(df_in_miss, ignore_index = True)
      df = df.append(df_out_miss, ignore_index = True)
      
      target_column = f"confidence for class {klass}"
      df_plot = df.rename(columns = { klass : target_column }) 

      sns.violinplot(x = 'type', y = target_column, 
                     data = df_plot[[target_column, 'training set', 'type']], 
                     scale = 'count', bw = 0.1, cut = 0)
      
      plt.savefig(save_path.as_posix() + "_all.pdf")
      plt.clf()
      
  def new_report_dir(self):
    dirs = [x for x in reports_path.iterdir() if reports_path.is_dir()]
    index = len(dirs)
    self.report_dir = reports_path/f"statistics_{index}"
    self.report_dir.mkdir()
  
  def membership_distributions(self, train_datasets, test_datasets):
    """process the mean distribution of all train and test MIA datasets 
    """
    dirs = [x for x in self.report_dir.iterdir() if self.report_dir.is_dir()]
    index = len(dirs)
    self.exp_dir = self.report_dir /f"experiment_{index}"
    self.exp_dir.mkdir()
    klass = 0
    for dataset in train_datasets:
      self._process_mia_dataset(dataset, klass, self.exp_dir/f"mia_train_samples_model_{klass}")
      klass += 1
    
    klass = 0
    for dataset in test_datasets:
      self._process_mia_dataset(dataset, klass, self.exp_dir/f"mia_test_samples_model_{klass}")
      klass += 1

  def _close_timer(self):
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
    self._call_stat("add_loss")
    self.exp[-1]['model_training'][-1]['loss'].append(loss)

  def _call_stat(self, method_name):
    pass
    # try:
    #   if 'number 6' in self.exp[-1]['name']:
    #     print(method_name, [ (model['name'],model['label']) for model in self.exp[-1]['model_training'] ])
    #     pdb.set_trace()
    # except:
    #   pass
