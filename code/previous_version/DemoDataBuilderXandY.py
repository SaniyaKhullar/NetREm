# DemoDataBuilder Class: :)
from packages_needed import *
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import plotly.express as px
class DemoDataBuilderXandY: 
    """:) Please note that this class focuses on building Y data based on a normal distribution (specified mean
    and standard deviation). M is the # of samples we want to generate. Thus, Y is a vector with M elements. 
    Then, this class returns X for a set of N predictors (each with M # of samples) based on a list of N correlation
    values. For instance, if N = 5 predictors (the Transcription Factors (TFs)), we have [X1, X2, X3, X4, X5],
    and a respective list of correlation values: [cor(X1, Y), cor(X2, Y), cor(X3, Y), cor(X4, Y), cor(X5, Y)].
    Then, this class will generate X, a matrix of those 5 predictors (based on similar distribution as Y) 
    with these respective correlations."""   
    
    _parameter_constraints = {
        "test_data_percent": (0, 100),
        "mu": (0, None),
        "std_dev": (0, None),
        "num_iters_to_generate_X": (1, None),
        "same_train_test_data": [False, True],
        "rng_seed": (0, None),
        "randSeed": (0, None),
        "ortho_scalar": (1, None),
        "orthogonal_X_bool": [True, False],
        "view_input_correlations_plot": [False, True],
        "num_samples_M": (1, None),
        "corrVals": list
    }
    
    def __init__(self, **kwargs):

        # define default values for constants
        self.same_train_test_data = False
        self.test_data_percent = 30
        self.mu = 0
        self.verbose = True
        self.std_dev = 1
        self.num_iters_to_generate_X = 100
        self.rng_seed = 2023 # for Y
        self.randSeed = 123 # for X
        self.orthogonal_X_bool = True # False adjustment made on 9/20
        self.ortho_scalar = 10
        self.tol = 1e-2
        self.view_input_correlations_plot = False
        # reading in user inputs
        self.__dict__.update(kwargs)
        ##################### other user parameters being loaded and checked
        self.same_train_and_test_data_bool = self.same_train_test_data
        # check that all required keys are present:
        required_keys = ["corrVals", "num_samples_M"]
        missing_keys = [key for key in required_keys if key not in self.__dict__]
        if missing_keys:
            raise ValueError(f":( Please note ye are missing information for these keys: {missing_keys}")
        self.M = self.num_samples_M
        self.N = self.get_N()
        self.y = self.generate_Y()
        self.X = self.generate_X()
        self.same_train_and_test_data_bool = self.same_train_test_data
        if self.same_train_and_test_data_bool:
            self.testing_size = 1
        else:
            self.testing_size = (self.test_data_percent/100.0)
        self.data_sets = self.generate_training_and_testing_data() # [X_train, X_test, y_train, y_test]
        self.X_train = self.data_sets[0]
        self.X_test = self.data_sets[1]
        self.y_train = self.data_sets[2]
        self.y_test = self.data_sets[3]
        
        self.tf_names_list = self.get_tf_names_list()
        self.corr_df = self.return_correlations_dataframe()
        self.combined_correlations_df = self.get_combined_correlations_df()
        if self.view_input_correlations_plot:
            self.view_input_correlations = self.view_input_correlations()
        self._apply_parameter_constraints()
        self.X_train_df = self.view_X_train_df()
        self.y_train_df = self.view_y_train_df()
        self.X_test_df = self.view_X_test_df()
        self.y_test_df = self.view_y_test_df()
        self.X_df = self.view_original_X_df()
        self.y_df = self.view_original_y_df()
        self.combined_train_test_x_and_y_df = self.combine_X_and_y_train_and_test_data()
        
    def _apply_parameter_constraints(self):
        constraints = {**DemoDataBuilderXandY._parameter_constraints}
        for key, value in self.__dict__.items():
            if key in constraints:
                if isinstance(constraints[key], tuple):
                    if isinstance(constraints[key][0], type) and not isinstance(value, constraints[key][0]):
                        setattr(self, key, constraints[key][0])
                    elif constraints[key][1] is not None and isinstance(constraints[key][1], type) and not isinstance(value, constraints[key][1]):
                        setattr(self, key, constraints[key][1])
                elif key == "corrVals": # special case for corrVals
                    if not isinstance(value, list):
                        setattr(self, key, constraints[key])
                elif value not in constraints[key]:
                    setattr(self, key, constraints[key][0])
        return self
        
    def get_tf_names_list(self):
        tf_names_list = []
        for i in range(0, self.N):
            term = "TF" + str(i+1)
            tf_names_list.append(term)
        return tf_names_list
    
    # getter method
    def get_N(self):
        N = len(self.corrVals)
        return N 
    
    def get_X_train(self):
        return self.data_sets[0] #X_train

    def get_y_train(self):
        return self.data_sets[2] # y_train
    
    def get_X_test(self):
        return self.data_sets[1]
    
    def get_y_test(self):       
        return self.data_sets[3]

    def view_original_X_df(self):
        import pandas as pd
        X_df = pd.DataFrame(self.X, columns = self.tf_names_list)
        return X_df
    
    def view_original_y_df(self):
        import pandas as pd
        y_df = pd.DataFrame(self.y, columns = ["y"])
        return y_df
    
    def view_X_train_df(self):
        import pandas as pd
        X_train_df = pd.DataFrame(self.X_train, columns = self.tf_names_list)
        return X_train_df

    def view_y_train_df(self):
        import pandas as pd
        y_train_df = pd.DataFrame(self.y_train, columns = ["y"])
        return y_train_df
    
    def view_X_test_df(self):
        X_test_df = pd.DataFrame(self.X_test, columns = self.tf_names_list)
        return X_test_df
    
    def view_y_test_df(self):
        y_test_df = pd.DataFrame(self.y_test, columns = ["y"])
        return y_test_df
    
    def combine_X_and_y_train_and_test_data(self):
        X_p1 = self.X_train_df
        X_p1["info"] = "training"
        X_p2 = self.X_test_df
        X_p2["info"] = "testing"
        X_combined = pd.concat([X_p1, X_p2]).drop_duplicates()
        y_p1 = self.y_train_df
        y_p1["info"] = "training"
        y_p2 = self.y_test_df
        y_p2["info"] = "testing"
        y_combined = pd.concat([y_p1, y_p2]).drop_duplicates()
        combining_df = X_combined
        combining_df["y"] = y_combined["y"]
        return combining_df

    def return_correlations_dataframe(self):
        corr_info = ["expected_correlations"] * self.N
        corr_df = pd.DataFrame(corr_info, columns = ["info"])
        corr_df["TF"] = self.tf_names_list
        corr_df["value"] = self.corrVals
        corr_df["data"] = "correlations"
        return corr_df
    
    def generate_Y(self):
        seed_val = self.rng_seed
        rng = np.random.default_rng(seed=seed_val)
        y = rng.normal(self.mu, self.std_dev, self.M)
        return y
    
        # Check if Q is orthogonal using the is_orthogonal function
    def is_orthogonal(matrix):
        """
        Checks if a given matrix is orthogonal.
        Parameters:
        matrix (numpy.ndarray): The matrix to check
        Returns:
        bool: True if the matrix is orthogonal, False otherwise.
        """
        # Compute the transpose of the matrix
        matrix_T = matrix.T

        # Compute the product of the matrix and its transpose
        matrix_matrix_T = np.dot(matrix, matrix_T)

        # Check if the product is equal to the identity matrix
        return np.allclose(matrix_matrix_T, np.eye(matrix.shape[0]))

#     # Define the modified generate_X function
#     def generate_X(self):
#         """Generates a design matrix X with the given correlations while introducing noise and dependencies.
#         Parameters:
#         orthogonal (bool): Whether to generate an orthogonal matrix (default=False).

#         Returns:
#         numpy.ndarray: The design matrix X.
#         """
#         orthogonal = self.orthogonal_X_bool
#         scalar = self.ortho_scalar
#         np.random.seed(self.randSeed)
#         y = self.y
#         n = len(y)
#         numTFs = self.N # len(corrVals)
#         numIterations = self.num_iters_to_generate_X
#         correlations = self.corrVals
#         corrVals = [correlations[0]] + correlations

#         # Step 1: Generate Initial X
#         e = np.random.normal(0, 1, (n, numTFs + 1))
#         X = np.copy(e)
#         X[:, 0] = y * np.sqrt(1.0 - corrVals[0]**2) / np.sqrt(1.0 - np.corrcoef(y, X[:,0])[0,1]**2)
#         for j in range(numIterations):
#             for i in range(1, numTFs + 1):
#                 corr = np.corrcoef(y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * y

#         # Step 2: Add Noise
#         noise_scale = 0.1  # You can adjust this value
#         X += np.random.normal(0, noise_scale, X.shape)

#         # Step 3: Introduce Inter-dependencies
#         # Make the second predictor a combination of the first and third predictors
#         X[:, 1] += 0.3 * X[:, 0] + 0.7 * X[:, 2]

#         # Step 4: Adjust for Correlations
#         for j in range(numIterations):
#             for i in range(1, numTFs + 1):
#                 corr = np.corrcoef(y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * y

#         if orthogonal:
#             # Compute the QR decomposition of X and take only the Q matrix
#             Q = np.linalg.qr(X)[0]
#             Q = scalar * Q
#             return Q[:, 1:]
#         else:
#             # Return the X matrix without orthogonalization
#             return X[:, 1:]

#     # # Display the modified function to ensure it looks okay
#     # print(generate_X_modified)

#     def generate_X(self):
#         """Generates a design matrix X with the given correlations and introduces an interaction term.

#         Returns:
#         numpy.ndarray: The design matrix X.
#         """
#         np.random.seed(self.randSeed)
#         y = self.y
#         n = len(y)
#         numTFs = self.N  # Number of predictors
#         numIterations = self.num_iters_to_generate_X
#         corrVals = self.corrVals

#         # Step 1: Generate Initial X based on the specified correlations with Y
#         e = np.random.normal(0, 1, (n, numTFs))
#         X = np.copy(e)
#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * y

#         # Step 2: Introduce Interaction Term into Y
#         interaction_term = X[:, 3] * X[:, 4]
#         self.y = y + 0.5 * interaction_term  # Adjust the coefficient as needed

#         # Step 3: Re-adjust for specified correlations with Y
#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(self.y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * self.y

#         return X



    # Define the modified generate_X function to highlight the benefits of network-regularized regression
#     def generate_X(self):
#         """Generates a design matrix X to highlight the benefits of network-regularized regression.

#         Returns:
#         numpy.ndarray: The design matrix X.
#         """
#         np.random.seed(self.randSeed)
#         n = len(self.y)
#         numTFs = self.N  # Number of predictors
#         numIterations = self.num_iters_to_generate_X
#         corrVals = self.corrVals

#         # Step 1: Generate Initial X based on the specified correlations with Y
#         e = np.random.normal(0, 1, (n, numTFs))
#         X = np.copy(e)
#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(self.y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * self.y

#         # Step 2: Weaken X2 and X4 as predictors by introducing interactions in Y
#         interaction_term = 0.3 * (X[:, 0] * X[:, 1]) + 0.3 * (X[:, 3] * X[:, 4])  # Interaction terms
#         self.y = self.y + interaction_term  # Update Y

#         # Step 3: Strengthen network edges by making X1 and X2, and X4 and X5 highly correlated
#         X[:, 1] = 0.7 * X[:, 0] + 0.3 * X[:, 1]  # X1 and X2
#         X[:, 3] = 0.7 * X[:, 4] + 0.3 * X[:, 3]  # X4 and X5

#         # Step 4: Re-adjust for specified correlations with Y
#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(self.y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * self.y

#         return X
#     def generate_X(self):
#         """Generates a design matrix X with the given correlations and introduces specified network edges and interactions.

#         Returns:
#         numpy.ndarray: The design matrix X.
#         """
#         np.random.seed(self.randSeed)
#         y = self.y
#         n = len(y)
#         numTFs = self.N  # Number of predictors
#         numIterations = self.num_iters_to_generate_X
#         corrVals = self.corrVals

#         # Step 1: Generate Initial X based on the specified correlations with Y
#         e = np.random.normal(0, 1, (n, numTFs))
#         X = np.copy(e)
#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * y

#         # Step 2: Weaken X2 and X4 as predictors by introducing interactions in Y
#         self.y = y + 0.3 * (X[:, 1] * X[:, 0]) + 0.3 * (X[:, 3] * X[:, 4])  # Adjust the coefficients as needed

#         # Step 3: Strengthen network edges by making X1 and X2, and X4 and X5 highly correlated
#         X[:, 1] = 0.7 * X[:, 0] + 0.3 * X[:, 1]  # X1 and X2
#         X[:, 3] = 0.7 * X[:, 4] + 0.3 * X[:, 3]  # X4 and X5

#         # Step 4: Re-adjust for specified correlations with Y
#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(self.y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * self.y

#         return X
#     def generate_X(self):
#         """Generates a design matrix X with given correlations and introduces inter-predictor correlations.

#         Returns:
#         numpy.ndarray: The design matrix X.
#         """
#         np.random.seed(self.randSeed)
#         y = self.y
#         n = len(y)
#         numTFs = self.N  # Number of predictors
#         numIterations = self.num_iters_to_generate_X
#         corrVals = self.corrVals

#         # Step 1: Generate Initial X based on the specified correlations with Y
#         e = np.random.normal(0, 1, (n, numTFs))
#         X = np.copy(e)
#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * y

#         # Step 2: Introduce Inter-predictor Correlations
#         # Make X1 and X2 highly correlated
#         X[:, 0] = 0.5 * X[:, 0] + 0.5 * X[:, 1]
#         # Make X4 and X5 highly correlated
#         X[:, 3] = 0.525 * X[:, 3] + 0.475 * X[:, 4]

#         # Step 3: Re-adjust for specified correlations with Y
#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * y

#         return X

#     def generate_X(self, tol=1e-4):
#         orthogonal = self.orthogonal_X_bool
#         scalar = self.ortho_scalar
#         np.random.seed(self.randSeed)
#         y = self.y
#         n = len(y)
#         numTFs = self.N

#         # Initialize X with standard normal distribution
#         X = np.random.normal(0, 1, (n, numTFs))

#         for i in range(numTFs):
#             desired_corr = self.corrVals[i]

#             while True:
#                 # Create a new predictor as a linear combination of original predictor and y
#                 X[:, i] = desired_corr * y + np.sqrt(1 - desired_corr ** 2) * X[:, i]

#                 # Standardize the predictor to have mean 0 and variance 1
#                 X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

#                 # Calculate the actual correlation
#                 actual_corr = np.corrcoef(y, X[:, i])[0, 1]

#                 # Calculate the difference between the actual and desired correlations
#                 diff = abs(actual_corr - desired_corr)

#                 if diff < tol:
#                     break

#         # Orthogonalize the predictors to make them independent of each other
#         Q, _ = np.linalg.qr(X)

#         if orthogonal:
#             # Scale the orthogonalized predictors
#             Q = scalar * Q
#             return Q
#         else:
#             # Return the orthogonalized predictors without scaling
#             return Q

#     def generate_X(self):
#         np.random.seed(self.randSeed)
#         y = self.y
#         n = len(y)
#         numTFs = self.N
#         tol = self.tol

#         # Initialize X with standard normal distribution (vectorized)
#         X = np.random.normal(0, 1, (n, numTFs))

#         # Standardize y for correlation calculation
#         y_std = (y - np.mean(y)) / np.std(y)

#         for i in tqdm(range(numTFs), desc="Generating predictors"):
#             desired_corr = self.corrVals[i]

#             while True:
#                 # Orthogonalize Xi against all previous predictors
#                 for j in range(i):
#                     coef = np.dot(X[:, i], X[:, j]) / np.dot(X[:, j], X[:, j])
#                     X[:, i] -= coef * X[:, j]

#                 # Create and standardize new predictor (vectorized)
#                 X[:, i] = desired_corr * y + np.sqrt(1 - desired_corr ** 2) * X[:, i]
#                 X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

#                 # Calculate actual correlation (vectorized)
#                 actual_corr = np.dot(y_std, X[:, i]) / n

#                 # Check if actual correlation is close enough to desired correlation
#                 if abs(actual_corr - desired_corr) < tol:
#                     break

#         # Orthogonalize X to reduce inter-predictor correlation (if required)
#         if self.orthogonal_X_bool:
#             X, _ = np.linalg.qr(X)

#         return X
    
    def generate_X(self):
        np.random.seed(self.randSeed)
        y = self.y
        n = len(y)
        numTFs = self.N
        tol = self.tol

        # Initialize X with standard normal distribution (vectorized)
        X = np.random.normal(0, 1, (n, numTFs))

        # Standardize y for correlation calculation
        y_std = (y - np.mean(y)) / np.std(y)

        for i in tqdm(range(numTFs), desc="Generating predictors"):
            desired_corr = self.corrVals[i]

            while True:
                # Create and standardize new predictor (vectorized)
                X[:, i] = desired_corr * y + np.sqrt(1 - desired_corr ** 2) * X[:, i]
                X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

                # Calculate actual correlation (vectorized)
                actual_corr = np.dot(y_std, X[:, i]) / n

                # Check if actual correlation is close enough to desired correlation
                if abs(actual_corr - desired_corr) < tol:
                    break

        # Orthogonalize X to reduce inter-predictor correlation (if required)
        if self.orthogonal_X_bool:
            X, _ = np.linalg.qr(X)

        return X
    def generate_X7(self):
        orthogonal = self.orthogonal_X_bool
        scalar = self.ortho_scalar
        np.random.seed(self.randSeed)
        y = self.y
        n = len(y)
        numTFs = self.N
        tol = self.tol

        # Initialize X with standard normal distribution
        X = np.random.normal(0, 1, (n, numTFs))

        desc_name = "Generating data for " + str(numTFs) + " Predictors with tolerance of " + str(tol) + " :) "
        for i in tqdm(range(numTFs), desc=desc_name):
            desired_corr = self.corrVals[i]

            while True:
                # Create a new predictor as a linear combination of original predictor and y
                X[:, i] = desired_corr * y + np.sqrt(1 - desired_corr ** 2) * X[:, i]

                # Standardize the predictor to have mean 0 and variance 1
                X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

                # Calculate the actual correlation
                actual_corr = np.corrcoef(y, X[:, i])[0, 1]

                # Calculate the difference between the actual and desired correlations
                diff = abs(actual_corr - desired_corr)

                if diff < tol:
                    break

        # Step 2: Orthogonalize the predictors to remove inter-predictor correlation
        X_ortho, _ = np.linalg.qr(X)
        
        # Step 3: Scale each orthogonalized predictor to match the desired correlation with y
        for i in tqdm(range(numTFs), desc="Rescaling orthogonalized predictors"):
            desired_corr = self.corrVals[i]
            
            while True:
                # Scale the orthogonalized predictor
                X_ortho[:, i] = desired_corr * y + np.sqrt(1 - desired_corr ** 2) * X_ortho[:, i]
                
                # Standardize the predictor
                X_ortho[:, i] = (X_ortho[:, i] - np.mean(X_ortho[:, i])) / np.std(X_ortho[:, i])
                
                # Calculate the actual correlation
                actual_corr = np.corrcoef(y, X_ortho[:, i])[0, 1]
                
                # Calculate the difference between the actual and desired correlations
                diff = abs(actual_corr - desired_corr)
                
                if diff < tol:
                    break

        if orthogonal:
            # Compute the QR decomposition of X and take only the Q matrix
            Q = np.linalg.qr(X_ortho)[0]
            Q = scalar * Q
            return Q
        else:
            # Return the X matrix without orthogonalization
            return X_ortho


    def generate_X5(self):
        orthogonal = self.orthogonal_X_bool
        scalar = self.ortho_scalar
        np.random.seed(self.randSeed)
        y = self.y
        n = len(y)
        numTFs = self.N
        tol = self.tol
        jitter = 0.05  # Noise level to reduce correlation between predictors

        # Initialize X with standard normal distribution
        X = np.random.normal(0, 1, (n, numTFs))

        desc_name = "Generating data for " + str(numTFs) + " Predictors with tolerance of " + str(tol) + " :) "
        for i in tqdm(range(numTFs), desc=desc_name):
            desired_corr = self.corrVals[i]

            while True:
                # Create a new predictor as a linear combination of original predictor and y
                X[:, i] = desired_corr * y + np.sqrt(1 - desired_corr ** 2) * X[:, i]

                # Add a small amount of noise to reduce correlation with other predictors
                X[:, i] += jitter * np.random.normal(0, 1, n)

                # Standardize the predictor to have mean 0 and variance 1
                X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

                # Calculate the actual correlation
                actual_corr = np.corrcoef(y, X[:, i])[0, 1]

                # Calculate the difference between the actual and desired correlations
                diff = abs(actual_corr - desired_corr)

                if diff < tol:
                    break

        if orthogonal:
            # Compute the QR decomposition of X and take only the Q matrix
            Q = np.linalg.qr(X)[0]
            Q = scalar * Q
            return Q
        else:
            # Return the X matrix without orthogonalization
            return X
        
    def generate_X3(self):
        orthogonal = self.orthogonal_X_bool
        scalar = self.ortho_scalar
        np.random.seed(self.randSeed)
        y = self.y
        n = len(y)
        numTFs = self.N
        tol = self.tol
        # Initialize X with standard normal distribution
        X = np.random.normal(0, 1, (n, numTFs))
        desc_name = "Generating data for " + str(numTFs) + " Predictors with tolerance of " + str(tol) + " :) "
        for i in tqdm(range(numTFs), desc=desc_name):
            desired_corr = self.corrVals[i]

            while True:
                # Create a new predictor as a linear combination of original predictor and y
                X[:, i] = desired_corr * y + np.sqrt(1 - desired_corr ** 2) * X[:, i]

                # Standardize the predictor to have mean 0 and variance 1
                X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

                # Calculate the actual correlation
                actual_corr = np.corrcoef(y, X[:, i])[0, 1]

                # Calculate the difference between the actual and desired correlations
                diff = abs(actual_corr - desired_corr)

                if diff < tol:
                    break

        if orthogonal:
            # Compute the QR decomposition of X and take only the Q matrix
            Q = np.linalg.qr(X)[0]
            Q = scalar * Q
            return Q
        else:
            # Return the X matrix without orthogonalization
            return X

    # Define the function for generating synthetic data with specific correlations and standard normal predictors
    def generate_X1(self):
        orthogonal = self.orthogonal_X_bool
        scalar = self.ortho_scalar
        np.random.seed(self.randSeed)
        y = self.y
        n = len(y)
        numTFs = self.N

        # Initialize X with standard normal distribution
        X = np.random.normal(0, 1, (n, numTFs))

        # Adjust X to achieve the desired correlations with y
        for i in range(numTFs):
            corr = self.corrVals[i]
            # Create a new predictor as a linear combination of original predictor and y
            X[:, i] = corr * y + np.sqrt(1 - corr ** 2) * X[:, i]

            # Standardize the predictor to have mean 0 and variance 1
            X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

        if orthogonal:
            # Compute the QR decomposition of X and take only the Q matrix
            Q = np.linalg.qr(X)[0]
            Q = scalar * Q
            return Q
        else:
            # Return the X matrix without orthogonalization
            return X
#     def generate_X(self):
#         orthogonal = self.orthogonal_X_bool
#         scalar = self.ortho_scalar
#         np.random.seed(self.randSeed)
#         y = self.y
#         n = len(y)
#         numTFs = self.N
#         numIterations = self.num_iters_to_generate_X
#         correlations = self.corrVals
#         corrVals = [correlations[0]] + correlations

#         # Initialize X with standard normal distribution
#         X = np.random.normal(0, 1, (n, numTFs))

#         for j in range(numIterations):
#             for i in range(numTFs):
#                 corr = np.corrcoef(y, X[:, i])[0, 1]
#                 X[:, i] = X[:, i] + (corrVals[i] - corr) * y
#                 # Standardize the predictor
#                 X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

#         if orthogonal:
#             # Compute the QR decomposition of X and take only the Q matrix
#             Q = np.linalg.qr(X)[0]
#             Q = scalar * Q
#             return Q
#         else:
#             # Return the X matrix without orthogonalization
#             return X


#     def generate_X(self):
#         orthogonal = self.orthogonal_X_bool
#         scalar = self.ortho_scalar
#         np.random.seed(self.randSeed)
#         y = self.y
#         n = len(y)
#         numTFs = self.N
#         tol=self.tol
#         # Initialize X with standard normal distribution
#         X = np.random.normal(0, 1, (n, numTFs))
#         numIterations = self.num_iters_to_generate_X
#         for iter_count in range(numIterations):
#             max_diff = 0  # Initialize maximum difference between actual and desired correlations for this iteration
#             for i in range(numTFs):
#                 desired_corr = self.corrVals[i]

#                 # Create a new predictor as a linear combination of original predictor and y
#                 X[:, i] = desired_corr * y + np.sqrt(1 - desired_corr ** 2) * X[:, i]

#                 # Standardize the predictor to have mean 0 and variance 1
#                 X[:, i] = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])

#                 # Calculate the actual correlation
#                 actual_corr = np.corrcoef(y, X[:, i])[0, 1]

#                 # Calculate the difference between the actual and desired correlations
#                 diff = abs(actual_corr - desired_corr)
#                 max_diff = max(max_diff, diff)

#             # If the maximum difference between actual and desired correlations is below the tolerance, break the loop
#             if max_diff < tol:
#                 break

#         if orthogonal:
#             # Compute the QR decomposition of X and take only the Q matrix
#             Q = np.linalg.qr(X)[0]
#             Q = scalar * Q
#             return Q
#         else:
#             # Return the X matrix without orthogonalization
#             return X

    def generate_X_old(self):
        """Generates a design matrix X with the given correlations.
        Parameters:
        orthogonal (bool): Whether to generate an orthogonal matrix (default=False).
        
        Returns:
        numpy.ndarray: The design matrix X.
        """
        orthogonal = self.orthogonal_X_bool
        scalar = self.ortho_scalar
        np.random.seed(self.randSeed)
        y = self.y
        n = len(y)
        numTFs = self.N # len(corrVals)
        numIterations = self.num_iters_to_generate_X
        correlations = self.corrVals
        corrVals = [correlations[0]] + correlations
        e = np.random.normal(0, 1, (n, numTFs + 1))
        X = np.copy(e)
        X[:, 0] = y * np.sqrt(1.0 - corrVals[0]**2) / np.sqrt(1.0 - np.corrcoef(y, X[:,0])[0,1]**2)
        for j in range(numIterations):
            for i in range(1, numTFs + 1):
                corr = np.corrcoef(y, X[:, i])[0, 1]
                X[:, i] = X[:, i] + (corrVals[i] - corr) * y
        
        if orthogonal:
            # Compute the QR decomposition of X and take only the Q matrix
            Q = np.linalg.qr(X)[0]
            Q = scalar * Q
            return Q[:, 1:]
        else:
            # Return the X matrix without orthogonalization
            return X[:, 1:]
       
    
    def generate_training_and_testing_data(self):
        same_train_and_test_data_bool = self.same_train_and_test_data_bool 
        X = self.X
        y = self.y
        if same_train_and_test_data_bool == False: # different training and testing datasets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.testing_size)
            if self.verbose:
                print(f"Please note that since we hold out {self.testing_size * 100.0}% of our {self.M} samples for testing, we have:")
                print(f"X_train = {X_train.shape[0]} rows (samples) and {X_train.shape[1]} columns (N = {self.N} predictors) for training.")
                print(f"X_test = {X_test.shape[0]} rows (samples) and {X_test.shape[1]} columns (N = {self.N} predictors) for testing.")
                print(f"y_train = {y_train.shape[0]} corresponding rows (samples) for training.")
                print(f"y_test = {y_test.shape[0]} corresponding rows (samples) for testing.")
        else: # training and testing datasets are the same :)
            X_train, X_test, y_train, y_test = X, X, y, y
            y_train = y
            y_test = y_train
            X_test = X_train
            if self.verbose:
                print(f"Please note that since we use the same data for training and for testing :) of our {self.M} samples. Thus, we have:")
                print(f"X_train = X_test = {X_train.shape[0]} rows (samples) and {X_train.shape[1]} columns (N = {self.N} predictors) for training and for testing")
                print(f"y_train = y_test = {y_train.shape[0]} corresponding rows (samples) for training and for testing.")    
        return [X_train, X_test, y_train, y_test]
    
    
    def get_combined_correlations_df(self):
        combined_correlations_df = self.actual_vs_expected_corrs_DefensiveProgramming_all_groups(self.X, self.y, 
                                                                                            self.X_train, 
                                                                                            self.y_train,
                                                                                        self.X_test, 
                                                                                            self.y_test,
                                                                                self.corrVals, 
                                                                                self.tf_names_list, 
                                                                             self.same_train_and_test_data_bool)
        return combined_correlations_df
    
    def actual_vs_expected_corrs_DefensiveProgramming_all_groups(self, X, y, X_train, y_train, X_test, y_test,
                                                                corrVals, tf_names_list,
                                                                 same_train_and_test_data_bool):
        overall_corrs_df = self.compare_actual_and_expected_correlations_DefensiveProgramming_one_data_group(X, y, corrVals, 
                                                                                         tf_names_list, same_train_and_test_data_bool, "Overall")
        training_corrs_df = self.compare_actual_and_expected_correlations_DefensiveProgramming_one_data_group(X_train, y_train, corrVals, 
                                                                                          tf_names_list, same_train_and_test_data_bool, "Training")
        testing_corrs_df = self.compare_actual_and_expected_correlations_DefensiveProgramming_one_data_group(X_test, y_test, corrVals, 
                                                                                         tf_names_list, same_train_and_test_data_bool, "Testing")
        combined_correlations_df = pd.concat([overall_corrs_df, training_corrs_df, testing_corrs_df]).drop_duplicates()
        return combined_correlations_df
    
    def compare_actual_and_expected_correlations_DefensiveProgramming_one_data_group(self, X_matrix, y, corrVals, 
                                                              predictor_names_list,
                                                              same_train_and_test_data_boolean,
                                                              data_type):
        # please note that this function by Saniya ensures that the actual and expected correlations are close
        # so that the simulation has the x-y correlations we were hoping for in corrVals
        updatedDF = pd.DataFrame(X_matrix)#.shape
        actualCorrsList = []
        for i in tqdm(range(0, len(corrVals))):
            expectedCor = corrVals[i]
            actualCor = np.corrcoef(updatedDF[i], y)[0][1]
            difference = abs(expectedCor - actualCor)
            predictor_name = predictor_names_list[i]
            actualCorrsList.append([i, predictor_name, expectedCor, actualCor, difference])
        comparisonDF = pd.DataFrame(actualCorrsList, columns = ["i", "predictor", "expected_corr_with_Y", "actual_corr", "difference"])
        comparisonDF["X_group"] = data_type
        num_samples = X_matrix.shape[0]
        if same_train_and_test_data_boolean:
            comparisonDF["num_samples"] = "same " + str(num_samples)
        else:
            comparisonDF["num_samples"] = "unique " + str(num_samples)
        return comparisonDF

        # Visualizing Functions :)
    def view_input_correlations(self):
        corr_val_df = pd.DataFrame(self.corrVals, columns = ["correlation"])#.transpose()
        corr_val_df.index = self.tf_names_list
        corr_val_df["TF"] = self.tf_names_list
        fig = px.bar(corr_val_df, x='TF', y='correlation', title = "Input Correlations for Dummy Example", barmode='group')
        fig.show()
        return fig
    
    
    def view_train_vs_test_data_for_predictor(self, predictor_name):
        combined_train_test_x_and_y_df = self.combined_train_test_x_and_y_df
        combined_correlations_df = self.combined_correlations_df
        print(combined_correlations_df[combined_correlations_df["predictor"] == predictor_name][["predictor", "actual_corr", "X_group", "num_samples"]])
        title_name = title = "Training Versus Testing Data Points for Predictor: " + predictor_name
        fig = px.scatter(combined_train_test_x_and_y_df, x=predictor_name, y="y", color = "info",
                        title = title_name)
        #fig.show()
        return fig
    
    
def generate_dummy_data(corrVals,
        num_samples_M = 10000,
        train_data_percent = 70,
        mu = 0,
        std_dev = 1,
        iters_to_generate_X = 100,
        orthogonal_X = False,
        ortho_scalar = 10,
        view_input_corrs_plot = False,
        verbose = True, rand_seed_x = 123, rand_seed_y = 2023):
    
    # the defaults
    same_train_test_data = False
    test_data_percent = 100 - train_data_percent
    if train_data_percent == 100: # since all of the data is used for training,
        # then the training and testing data will be the same :)
        same_train_test_data = True
        test_data_percent = 100
    print(f":) same_train_test_data = {same_train_test_data}")
    demo_dict = {
        "test_data_percent": 100 - train_data_percent,
        "mu": mu, "std_dev": std_dev,
        "num_iters_to_generate_X": iters_to_generate_X,
        "same_train_test_data": same_train_test_data,
        "rng_seed": rand_seed_y, #2023, # for Y
        "randSeed": rand_seed_x, #123, # for X
        "ortho_scalar": ortho_scalar,
        "orthogonal_X_bool": orthogonal_X,
        "view_input_correlations_plot": view_input_corrs_plot,
        "num_samples_M": num_samples_M,
        "corrVals": corrVals, "verbose":verbose}
    dummy_data = DemoDataBuilderXandY(**demo_dict) # 
    return dummy_data
