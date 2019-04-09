from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, roc_auc_score, classification_report

class Statistics:
  def __init__(self):
    self.y_pred = []
    self.y_true = []
    self.balanced_accuracy_score = {}
    self.list_experiment = []
    
  def new_experiment(self, name, parameters):
    self.current_experiment_name = name
    self.list_experiment.append([self.current_experiment_name, parameters])
    self.param[self.current_experiment_name] = parameters
    self.balanced_accuracy_score[self.current_experiment_name] = []
    self.iteration = 0
    return
    
  def new_test_epoch():
    self.iteration += 1
    
    return
    
  def add_batch_results(self, batch_pred, batch_true):
    self.y_pred += batch_pred
    self.y_true += batch_true
    self.balanced_accuracy[self.current_experiment_name].append(balanced_accuracy_score(self.y_pred, self.y_true))
    
  def finalize(self):
    self.confusion_matrix = confusion_matrix(self.y_pred, self.y_true)
    self.roc_area = roc_auc_score(self.y_pred, self.y_true)
    self.report = classification_report(self.y_pred, self.y_true)
    self.y_pred = []
    self.y_true = []
    self.iteration = 0

    
  def save(self, file_path):
    return
    
  def print_results(self):
    self.finalize()
    print("Balanced accuracy score: {}".format(self.balanced_accuracy))
    print("Confusion matrix:\n {}".format(self.confusion_matrix))
    print("Curve ROC area: {}".format(self.roc_area))
    print(self.report)
    
    
  def plot(self):
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
