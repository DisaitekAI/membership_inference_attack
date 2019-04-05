from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

class Statistics:
  def __init__(self):
    self.y_pred = []
    self.y_true = []
    
  def new_experiment(self, name, parameters):
    return
    
  def new_test_epoch():
    return
    
  def add_batch_results(self, batch_pred, batch_true):
    self.y_pred += batch_pred
    self.y_true += batch_true
    
  def finalize(self):
    self.balanced_accuracy = balanced_accuracy_score(self.y_pred, self.y_true)
    self.y_pred = []
    self.y_true = []
    
  def save(self, file_path):
    return
    
  def print_results(self):
    self.finalize()
    print("balanced accuracy score: {}".format(self.balanced_accuracy))
    
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
