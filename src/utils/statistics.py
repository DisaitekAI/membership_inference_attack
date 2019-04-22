from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

class Statistics:
  def __init__(self):
    self.y_pred = []
    self.y_true = []
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
    pass

  def new_experiment(self, name, parameters):
    self.process_batchs()
    
    # ~ self.current_experiment_name = name
    # ~ self.list_experiment.append([self.current_experiment_name, parameters])
    # ~ self.param[self.current_experiment_name] = parameters
    # ~ self.balanced_accuracy[self.current_experiment_name] = []
    # ~ self.roc_area[self.current_experiment_name] = []
    # ~ self.batch_count[self.current_experiment_name] = 0
    # ~ self.epoch_count[self.current_experiment_name] = 0

  def new_epoch(self):
    self.process_batchs()
    # ~ self.epoch_count[self.current_experiment_name] += 1
    pass
        
  def new_batch(self, batch_pred, batch_true):
    # ~ if (self.epoch_count[self.current_experiment_name] == 1):
        # ~ self.batch_count[self.current_experiment_name] += 1
    self.y_pred += batch_pred
    self.y_true += batch_true
    # ~ self.balanced_accuracy[self.current_experiment_name].append(balanced_accuracy_score(self.y_pred, self.y_true))
    # ~ self.roc_area[self.current_experiment_name].append(roc_auc_score(self.y_pred, self.y_true))
    pass
    
  # ~ def end_epoch(self):
    # ~ self.confusion_matrix = confusion_matrix(self.y_pred, self.y_true)
    # ~ self.report = classification_report(self.y_pred, self.y_true)
    # ~ self.y_pred = []
    # ~ self.y_true = []
    # ~ self.iteration = 0

  def save(self, log_dir): # à appeler à la toute fin
    self.process_batchs()
    pass
    
  def print_results(self): # à appeler à chaque fin d'epoch
    self.process_batchs()
    
    print(f"Balanced accuracy score: {balanced_accuracy_score(self.y_pred, self.y_true)}")
    print(classification_report(self.y_pred, self.y_true))
    print(confusion_matrix(self.y_pred, self.y_true))
    self.y_pred = []
    self.y_true = []
    # ~ self.end_epoch()
    # ~ print("Balanced accuracy score: {}".format(self.balanced_accuracy[self.current_experiment_name][-1]))
    # ~ print("Confusion matrix:\n {}".format(self.confusion_matrix))
    # ~ print("Curve ROC area: {}".format(self.roc_area))
    # ~ print(self.report)
    pass
    
  def plot(self):
    print(self.batch_count)
    print(self.epoch_count)
    fig, ax1 = plt.subplots(1,1,figsize=(5,3),dpi=100)
    ax1.plot(self.balanced_accuracy[self.current_experiment_name], )
    #ax1.set_xlim((self.balanced_accuracy[self.current_experiment_name][0],self.balanced_accuracy[self.current_experiment_name][-1]))
    print(self.balanced_accuracy[self.current_experiment_name])
    xticklabs = ["{}".format(i) for i in range(self.epoch_count[self.current_experiment_name] * self.batch_count[self.current_experiment_name])]
    for i in range(self.epoch_count[self.current_experiment_name]):
        xticklabs[i * self.batch_count[self.current_experiment_name] - 1] = "Epoch {}".format(i) #r'$\textcolor{{red}}{{Epoch {}}}$'
    xticklabs[-1] = "Epoch {}".format(self.epoch_count[self.current_experiment_name])
    ax1.set_xticklabels(xticklabs)
    print(xticklabs)
    ax1.xaxis.set_minor_locator(MultipleLocator)
    #for i in range(self.epoch_count):
         #, data=(np.array([j for j in range(3)]),np.array([j for j in range(3)])
        #plt.title('Epoch {}'.format(self.iteration))
        #plt.axvline(x=i*self.epoch_count,color='red')
    #plt.show()
    fig.tight_layout()

  def new_train(self, name = None, label = None):
    """
    define a new model training.
    
    :name name of the training. 
    :label group model training with the same label. 
    """
    self.process_batchs()
  
    
  def new_epoch(self):
    """
    define a new epoch
    """
    self.process_batchs()
  
  def process_batchs(self):
    # ~ if len(batch_results) != 0:
      # calculer les metrics

    # ~ batch_results = []
    pass
    

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

