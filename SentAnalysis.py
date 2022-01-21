import tensorflow as tf
import lcrModel
import lcrModelInverse
import lcrModelAlt
import cabascModel
import svmModel
from OntologyReasonerIsik import OntReasonerFuck
from loadData import *

#import parameter configuration and data paths
from config import *

#import modules
import numpy as np
import sys

loadData = False
useOntology = False
runCABASC = False
runLCRROT = False
runLCRROTINVERSE = False
runLCRROTALT = True
runSVM = False

weightanalysis = False

# determine if backupmethod is used
if runCABASC or runLCRROT or runLCRROTALT or runLCRROTINVERSE or runSVM:
    backup = True
else:
    backup = False

# retrieve data and wordembeddings
#train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
#print(test_size)
remaining_size = 250
accuracyOnt = 0.87

print(FLAGS.test_path)
backup = True
print('Starting Ontology Reasoner')
Ontology = OntReasonerFuck()
#out of sample accuracy
accuracyOnt, remaining_size = Ontology.run(backup,FLAGS.test_path, runSVM)
