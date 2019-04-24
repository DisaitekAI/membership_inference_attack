from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pdb
import os, pathlib, sys

class Statistics:
      def __init__(self):
        self.y_pred = []
        self.y_true = []
        self.exp = []
        # ~ self.balanced_accuracy = dict()
        # ~ self.list_experiment = []
        # ~ self.current_experiment_name = ""
        # ~ self.list_experiment = []
        # ~ self.param = dict()
        # ~ self.confusion_matrix = [[0,0],[0,0]]
        # ~ self.roc_area = dict()
        # ~ self.report = ""
        # ~ self.batch_count = dict()
        # ~ self.epoch_count = dict()
    

      def new_experiment(self, name, parameters):
        self.process_batchs()
        experiment = {"name": name, "param": parameters, "model_training": []}
        self.exp.append(experiment)
        # ~ self.current_experiment_name = name
        # ~ self.list_experiment.append([self.current_experiment_name, parameters])
        # ~ self.param[self.current_experiment_name] = parameters
        # ~ self.balanced_accuracy[self.current_experiment_name] = []
        # ~ self.roc_area[self.current_experiment_name] = []
        # ~ self.batch_count[self.current_experiment_name] = 0
        # ~ self.epoch_count[self.current_experiment_name] = 0

      def new_train(self, name = None, label = None):
        """
        define a new model training.
    
        :name name of the training. 
        :label group model training with the same label. 
        """
        self.process_batchs()
        model = {"name": name, "label": label, "measures": {"balanced_accuracy": []} }
        self.exp[-1]["model_training"].append(model)

      def new_epoch(self):
        self.process_batchs()
        # ~ self.epoch_count[self.current_experiment_name] += 1
        
      def new_batch(self, batch_pred, batch_true):
        # ~ if (self.epoch_count[self.current_experiment_name] == 1):
            # ~ self.batch_count[self.current_experiment_name] += 1
        self.y_pred.extend(batch_pred)
        self.y_true.extend(batch_true)
        # ~ self.balanced_accuracy[self.current_experiment_name].append(balanced_accuracy_score(self.y_pred, self.y_true))
        # ~ self.roc_area[self.current_experiment_name].append(roc_auc_score(self.y_pred, self.y_true))
    
  # ~ def end_epoch(self):
    # ~ self.confusion_matrix = confusion_matrix(self.y_pred, self.y_true)
    # ~ self.report = classification_report(self.y_pred, self.y_true)
    # ~ self.y_pred = []
    # ~ self.y_true = []
    # ~ self.iteration = 0

      def process_batchs(self):
        if len(self.y_true) != 0:
            accuracy = balanced_accuracy_score(self.y_pred, self.y_true)
            self.exp[-1]["model_training"][-1]["measures"]["balanced_accuracy"].append(accuracy)
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
                        plot_path = path/experiment["name"]/model["name"]/measure_name
                        os.makedirs(os.path.dirname(str(plot_path)), exist_ok=True)
                        plt.plot(measure_values)
                        plt.title(measure_name)
                        plt.savefig(str(plot_path))

    
      def print_results(self):
        self.process_batchs()

        for experiment in self.exp:
            print("\n   Experiment " + experiment["name"] + " :\n")
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

            for group_label, group in groups.items():
                print("\nAverage statistics of the " + group_label + " models :\n")
                for measure_name in group[0]["measures"]: # every models with same label must have same measures
                        values = [model["measures"][measure_name] for model in group]
                        average = None
                        if type(values[0]) == float:
                            average = sum(values)/len(values)
                        if type(values[0]) == list:
                            final_values = [l[-1] for l in values]
                            average = sum(final_values) / len(final_values)
                            #average = [sum(x) / len(x) for x in zip(*values)]
                        if average != None:
                            print("\n" + measure_name + " : ")
                            print(average)
                            print("\n")

        # ~ print("Balanced accuracy score: {balanced_accuracy_score(self.y_pred, self.y_true)}")
        # ~ print(classification_report(self.y_pred, self.y_true))
        # ~ print(confusion_matrix(self.y_pred, self.y_true))
        # ~ self.y_pred = []
        # ~ self.y_true = []
        # ~ self.end_epoch()
        # ~ print("Balanced accuracy score: {}".format(self.balanced_accuracy[self.current_experiment_name][-1]))
        # ~ print("Confusion matrix:\n {}".format(self.confusion_matrix))
        # ~ print("Curve ROC area: {}".format(self.roc_area))
        # ~ print(self.report)
        
    
      def plot(self, val):
        fig, ax1 = plt.subplots(1,1,figsize=(5,3),dpi=100)
        ax1.plot(val)
        #ax1.set_xlim((self.balanced_accuracy[self.current_experiment_name][0],self.balanced_accuracy[self.current_experiment_name][-1]))
        # xticklabs = ["{}".format(i) for i in range(self.epoch_count[self.current_experiment_name] * self.batch_count[self.current_experiment_name])]
        # for i in range(self.epoch_count[self.current_experiment_name]):
        #     xticklabs[i * self.batch_count[self.current_experiment_name] - 1] = "Epoch {}".format(i) #r'$\textcolor{{red}}{{Epoch {}}}$'
        # xticklabs[-1] = "Epoch {}".format(self.epoch_count[self.current_experiment_name])
        # ax1.set_xticklabels(xticklabs)
        #ax1.xaxis.set_minor_locator(MultipleLocator)
        #for i in range(self.epoch_count):
             #, data=(np.array([j for j in range(3)]),np.array([j for j in range(3)])
            #plt.title('Epoch {}'.format(self.iteration))
            #plt.axvline(x=i*self.epoch_count,color='red')
        #plt.show()
        fig.tight_layout()
  
    

# ~ new_experiment
  # ~ -> new_train(soit label, soit name, soit les deux)
      # ~ -> new_epoch
          # ~ -> new_batch(results)
            # ~ batchs.append(results)
          
  # ~ -> new_train
    # ~ ...
    
# ~ new_experiment
  # ~ ....
  
# ~ print_results
# ~ save

# ~ 1 - ecrire la classe statistics pour la balanced accuracy score (le tester avec le code)
# ~ 2 - écrire ou update les tests unitaires
# ~ 3 - ajouter une measure (tester)
# ~ 4 - update les tests unitaires
# ~ 5 - go to 3

# ~ Data de la classe stats : 
  # ~ exps = [ experiment1, experiment2, ..... ]
    # ~ -> experiment
        # ~ -> name
        # ~ -> params
        # ~ -> model_training = [ model1, model2 .... ]
            # ~ -> model
              # ~ -> name or None
              # ~ -> label or None
              # ~ -> measures = { name_measure1 : value, name_measure2 : value , .... }

