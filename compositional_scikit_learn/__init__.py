__version__ = '0.0.1'

import logging
logger = logging.getLogger(__name__)

from .transformers import LRTransorm
from .regression import CompositionalLinearRegression, CompositionalRidge, CompositionalLasso, CompositionalSVR 
from .regression import CompositionalElasticNet, CompositionalDecisionTreeRegressor, CompositionalRandomForestRegressor 
from .regression import CompositionalGradientBoostingRegressor, CompositionalKNeighborsRegressor
from .classification import CompositionalLogisticRegression, CompositionalSVC, CompositionalDecisionTreeClassifier, CompositionalSGDClassifier
from .classification import CompositionalRandomForestClassifier, CompositionalGradientBoostingClassifier, CompositionalKNeighborsClassifier
from .clusterization import CompositionalAgglomerativeClustering, CompositionalDBSCAN, CompositionalKMeans
from .utils import _apply_transform, softmax
