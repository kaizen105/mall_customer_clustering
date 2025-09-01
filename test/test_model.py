import unittest
import joblib
from sklearn.cluster import KMeans

class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        model = joblib.load('model/customer_segmentation.pkl')
        self.assertIsInstance(model, KMeans)
        self.assertEqual(model.n_clusters, 5)  # check cluster count

if __name__ == '__main__':
    unittest.main()
