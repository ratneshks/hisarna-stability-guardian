import unittest
import numpy as np
import pandas as pd
from data.generator import HIsarnaDataGenerator
from model.preprocessing import HIsarnaPreprocessor

class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = HIsarnaDataGenerator(seed=42)

    def test_initial_state(self):
        reading = self.generator.get_latest_reading()
        self.assertEqual(len(reading), 10)
        self.assertIn('T_cyclone', reading)
        self.assertEqual(self.generator.mode, 'stable')

    def test_get_history(self):
        # Generate some readings manually to populate history
        for _ in range(5):
            self.generator._generate_step()
            
        history = self.generator.get_history(5)
        self.assertIsInstance(history, pd.DataFrame)
        self.assertEqual(len(history), 5)
        self.assertEqual(history.shape[1], 10)

class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.generator = HIsarnaDataGenerator(seed=42)
        # Fast-forward to gather warmup data
        warmup_data = []
        for _ in range(300):
            self.generator._generate_step()
            warmup_data.append(self.generator.get_latest_reading())
        self.df_warmup = pd.DataFrame(warmup_data)
        self.preprocessor = HIsarnaPreprocessor()

    def test_normalization_fit(self):
        self.preprocessor.fit(self.df_warmup)
        self.assertTrue(hasattr(self.preprocessor, 'mu_dict'))
        self.assertTrue(hasattr(self.preprocessor, 'sigma_dict'))
        self.assertEqual(len(self.preprocessor.mu_dict), 10)

    def test_feature_engineering_and_windowing(self):
        self.preprocessor.fit(self.df_warmup)
        # Take the last 30 readings as a window
        window_df = self.df_warmup.iloc[-30:].copy()
        
        # process_window should return a 308-dim vector
        tensor_input = self.preprocessor.process_window(window_df, current_time=0.0)
        import torch
        self.assertIsInstance(tensor_input, torch.Tensor)
        self.assertEqual(tensor_input.shape, (308,))

if __name__ == '__main__':
    unittest.main()
