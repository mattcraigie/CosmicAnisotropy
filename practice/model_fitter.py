import matplotlib.pyplot as plt
import numpy as np

import model_maker
import data_maker

flat_lcdm_params = (67.7, 0.31, 0.69)

model = model_maker.basic_model
data = data_maker.get_data(model, flat_lcdm_params)

omega_m_range = np.linspace(0.2, 0.4, 10)
omega_lambda_range = np.linspace(0.6, 0.8, 10)
H0_range = np.linspace(60, 80, 10)

combinations = np.array(np.meshgrid(omega_m_range, omega_lambda_range, H0_range)).T.reshape(-1, 3)

for i in range(np.shape(combinations)[0]):
    


