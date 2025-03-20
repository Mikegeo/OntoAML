import shap


class ShapValuesBuilder:

    @staticmethod
    def create_shap_explainer(model, explainer_type='auto'):
        shap_explainer = ""
        if explainer_type == 'auto':
            shap_explainer = shap.Explainer(model)
        elif explainer_type == 'tree':
            shap_explainer = shap.TreeExplainer(model)
        # need to define the data
        # elif explainer_type == 'kernel':
        #     shap_explainer = shap.KernelExplainer(model)
        # elif explainer_type == 'linear':
        #     shap_explainer = shap.LinearExplainer(model)
        return shap_explainer

    
