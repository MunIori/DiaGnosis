import pandas as pd # Para la lectura del corpus
from sklearn.model_selection import train_test_split # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import KFold # Para dividir los datos de entrenamiento en conjuntos de validacion
import matplotlib.pyplot as plt # Para graficar los datos
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report # Para las metricas
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle

import warnings

warnings.filterwarnings("ignore")

NUMERO_PLIEGUES = 10

# Define un conjunto de validacion
class validation_set:
  def __init__(self, X_train, y_train, X_test, y_test):
    self.X_train = X_train
    self.y_train = y_train
    self.X_test = X_test
    self.y_test = y_test

# Define un conjunto de entrenamiento
class train_set:
  def __init__(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

# Define un conjunto de prueba
class test_set:
  def __init__(self, X_test, y_test):
    self.X_test = X_test
    self.y_test = y_test

# Define un conjunto de datos
class data_set:
  def __init__(self, validation_set, train_set, test_set):
    self.validation_set = validation_set
    self.train_set = train_set
    self.test_set = test_set

# Divide el conjunto de datos en entrenamiento, prueba y validacion
def generate_train_test(df, pliegues):    
  # Lee el corpus original del archivo de entrada y lo pasa a un DataFrame
  X = df.drop(['Outcome'],axis=1).values   
  y = df['Outcome'].values

  # Separa corpus en conjunto de entrenamiento y prueba
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)    
  # Crea pliegues para la validación cruzada
  validation_sets = []
  kf = KFold(n_splits=pliegues)
  for train_index, test_index in kf.split(X_train):
    X_train_v, X_test_v = X_train[train_index], X_train[test_index]
    y_train_v, y_test_v = y_train[train_index], y_train[test_index]
    # Agrega el pliegue creado a la lista
    validation_sets.append(validation_set(X_train_v, y_train_v, X_test_v, y_test_v))
  # Almacena el conjunto de entrenamiento
  my_train_set = train_set(X_train, y_train)    
  # Almacena el conjunto de prueba
  my_test_set = test_set(X_test, y_test)    
  # Guarda el dataset con los pliegues del conjunto de validación y el conjunto de pruebas
  my_data_set = data_set(validation_sets, my_train_set, my_test_set)
  return (my_data_set)

def pruebas(datos,target_name,clasificador,bayes=False):
  accuracy_pliegues = []
  f1_pligues = []
  recall_pliegues = []
  precision_pligues = []
  for i in range(NUMERO_PLIEGUES):

    y_pliegue_entrenamiento =datos.validation_set[i].y_train
    y_pliegue_prueba = datos.validation_set[i].y_test
    X_pliegue_entrenamiento = datos.validation_set[i].X_train
    X_pliegue_prueba = datos.validation_set[i].X_test

    clf = clasificador

    if(bayes):
        print("MINMAX------------------------------")
        scaler = MinMaxScaler()
        X_pliegue_entrenamiento = scaler.fit_transform(X_pliegue_entrenamiento)
    
    clf.fit(X_pliegue_entrenamiento, y_pliegue_entrenamiento)

    y_pred = clf.predict(X_pliegue_prueba)

    metricas = classification_report(y_pliegue_prueba, y_pred, target_names=target_name, output_dict=True, zero_division=0)
    print(classification_report(y_pliegue_prueba, y_pred, target_names=target_name, output_dict=False, zero_division=0))
    accuracy_pliegues.append(metricas["accuracy"])
    promedio_recall = 0
    promedio_f1 = 0
    promedio_precision = 0
    for clase in target_name:
        promedio_recall += metricas[clase]["recall"]
        promedio_f1 += metricas[clase]["f1-score"]
        promedio_precision += metricas[clase]["precision"]

    recall_pliegues.append(promedio_recall/len(target_name))
    
    f1_pligues.append(promedio_f1/len(target_name))
    
    precision_pligues.append(promedio_precision/len(target_name))
  
  accuracy_pliegues = sum(accuracy_pliegues) / NUMERO_PLIEGUES
  f1_pliegues = sum(f1_pligues) / NUMERO_PLIEGUES
  recall_pliegues = sum(recall_pliegues) / NUMERO_PLIEGUES
  precision_pliegues = sum(precision_pligues) / NUMERO_PLIEGUES

  print(f"Accuracy promedio: {accuracy_pliegues}")
  print(f"f1 promedio: {f1_pliegues}")
  print(f"Recall promedio: {recall_pliegues}")
  print(f"precision promedio: {precision_pliegues}")

def clasificador_datos_pruebas(datos,target_name,clasificador,bayes=False):

    y_entrenamiento = datos.train_set.y_train
    y_prueba = datos.test_set.y_test
    X_entrenamiento = datos.train_set.X_train
    X_prueba = datos.test_set.X_test

    clf = clasificador
    if(bayes):
        scaler = MinMaxScaler()
        X_entrenamiento = scaler.fit_transform(X_entrenamiento)

    clf.fit(X_entrenamiento, y_entrenamiento)

    filename = 'Final_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

    y_pred = clf.predict(X_prueba) #la clase predicha

    cm = confusion_matrix(y_prueba, y_pred,labels= clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negativo","Positivo"])
    disp.plot()

    metricas = classification_report(y_prueba, y_pred, target_names=target_name, output_dict=True, zero_division=0)
    print(classification_report(y_prueba, y_pred, target_names=target_name, output_dict=False, zero_division=0))
    promedio_recall = 0
    promedio_f1 = 0
    promedio_precision = 0
    for clase in target_name:
        promedio_recall += metricas[clase]["recall"]
        promedio_f1 += metricas[clase]["f1-score"]
        promedio_precision += metricas[clase]["precision"]
    promedio_recall /= len(target_name)
    promedio_f1 /= len(target_name)
    promedio_precision /= len(target_name)


    print(f"Accuracy promedio: {metricas['accuracy']}")
    print(f"f1 promedio: {promedio_f1}")
    print(f"Recall promedio: {promedio_recall}")
    print(f"precision promedio: {promedio_precision}")

def main():
  
  df = pd.read_csv('diabetes.csv', sep=',', engine='python')

  df.drop('DiabetesPedigreeFunction', inplace=True, axis=1)

  datos = generate_train_test(df, NUMERO_PLIEGUES)

  clasificador_logistico = LogisticRegression()

  target_names = ["0","1"]

  print("Logistico")
  pruebas(datos,target_names,clasificador_logistico)
  clasificador_datos_pruebas(datos,target_names,clasificador_logistico)

  plt.show()

if __name__ == '__main__':
  
  main()