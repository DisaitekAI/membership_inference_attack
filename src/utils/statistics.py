from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
import numpy as np

class Statistics:
  def __init__(self):
    self.y_pred = []
    self.y_true = []
    self.balanced_accuracy = dict()
    self.list_experiment = []
    
    self.current_experiment_name = ""
    self.list_experiment = []
    self.param = dict()
    self.iteration = 0
    self.confusion_matrix = [[0,0],[0,0]]
    self.roc_area = dict()
    self.report = ""
    self.batch_count = 0
    self.epoch_count = 0


    
    ############# Entrées #############
    
  def new_experiment(self, name, parameters):
    self.current_experiment_name = name
    self.list_experiment.append([self.current_experiment_name, parameters])
    self.param[self.current_experiment_name] = parameters
    self.balanced_accuracy[self.current_experiment_name] = []
    self.roc_area[self.current_experiment_name] = []
    self.iteration = 0
    self.batch_count = 0
    self.epoch_count = 0

    
  def new_epoch(self):
    self.iteration += 1
        
  def add_batch_results(self, batch_pred, batch_true):
    self.batch_count += 1
    self.y_pred += batch_pred
    self.y_true += batch_true
    self.balanced_accuracy[self.current_experiment_name].append(balanced_accuracy_score(self.y_pred, self.y_true))
    self.roc_area[self.current_experiment_name].append(roc_auc_score(self.y_pred, self.y_true))
    
  def end_epoch(self):
    self.epoch_count = self.iteration
    self.confusion_matrix = confusion_matrix(self.y_pred, self.y_true)
    self.report = classification_report(self.y_pred, self.y_true)
    self.y_pred = []
    self.y_true = []
    self.iteration = 0

    ############# Sorties #############

    
  def save(self, file_path): # à appeler à la toute fin
    return
    
  def print_results(self): # à appeler à chaque fin d'epoch
    self.end_epoch()
    print("Balanced accuracy score: {}".format(self.balanced_accuracy[self.current_experiment_name][-1]))
    print("Confusion matrix:\n {}".format(self.confusion_matrix))
    print("Curve ROC area: {}".format(self.roc_area))
    print(self.report)
    
    
  def plot(self):
    print(self.batch_count)
    print(self.epoch_count)
    plt.plot(self.balanced_accuracy[self.current_experiment_name])
    x = self.balanced_accuracy[self.current_experiment_name]
    for i in range(self.epoch_count):
         #, data=(np.array([j for j in range(3)]),np.array([j for j in range(3)])
        plt.title('Epoch {}'.format(self.iteration))
        plt.axvline(x=i*self.epoch_count,color='red')
    plt.show()
    return
    
# ~ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
# ~ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
# ~ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
# ~ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
# ~ https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

# ~ Plusieurs experiences
  # ~ pour chaque experience
    # ~ plusieurs fonction de test (une par epoch d'entrainement)
      # ~ pour chaque test tu as plusieurs résultats de batch
        # ~ toi tu reçois les résultats d'un batch 
