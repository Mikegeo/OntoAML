import joblib
import pandas as pd
import numpy as np


class PipelineLoader:

    @staticmethod
    def load_pipeline(pipeline_path):
        pipeline = joblib.load(pipeline_path)
        return pipeline

    @staticmethod
    def load_preprocessor(preprocessor_path):
        preprocessor = joblib.load(preprocessor_path)
        return preprocessor

    @staticmethod
    def get_model_from_pipeline(pipeline, ml_task):
        if ml_task == 'classification':
            model = pipeline.named_steps['classifier']
        else:
            model = pipeline.named_steps['regressor']
        return model

    @staticmethod
    def get_preprocessor_from_pipeline(pipeline):
        preprocessor = pipeline.named_steps['preprocessor']
        return preprocessor

    @staticmethod
    def get_pca_from_pipeline(pipeline):
        pca = pipeline.named_steps['pca']
        return pca

    @staticmethod
    def get_features_names_from_preprocessor(preprocessor):
        feature_names = preprocessor.get_feature_names_out()
        return feature_names

    @staticmethod
    def get_top_features_from_pca(pca, feature_names):
        components = pca.components_
        components_df = pd.DataFrame(components, columns=feature_names)
        # Calculate the contribution of each feature to each principal component
        contributions = np.abs(components_df)
        # Get the overall importance of each feature by summing contributions across all components
        feature_importance = contributions.sum(axis=0)
        # Sort features by their importance
        sorted_features = feature_importance.sort_values(ascending=False)
        # Retrieve the list of the top features (for example, top 10 features)
        top_features = sorted_features.head(pca.n_components).index.tolist()
        return top_features
