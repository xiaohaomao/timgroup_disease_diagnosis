import os
from core.explainer.predict_explainer import SingleResultExplainer, MultiModelExplainer
class Predictor(object):
	def __init__(self):
		pass


class MultiModelPredictor(Predictor):
	def __init__(self):
		super(MultiModelPredictor, self).__init__()
		self.model_dict = {}
		self.pa_hpo_list = None
		self.text_searcher = None
		self.TOP_N = 20


	def put_model(self, key, model):
		self.model_dict[key] = model


	def set_text_searcher(self, text_searcher):
		self.text_searcher = text_searcher


	def set_patient_from_text(self, pa_text):
		self.pa_hpo_list, _ = self.text_searcher.search(pa_text)


	def set_patient_from_hpo_list(self, pa_hpo_list):
		self.pa_hpo_list = pa_hpo_list


	def predict(self, model_key):
		model = self.model_dict[model_key]
		return model.query(self.pa_hpo_list, topk=None)


	def predict_many(self):
		return {model_key:self.predict(model_key) for model_key in self.model_dict}


	def predict_explain_save(self, folder, diag_list=None):
		os.makedirs(folder, exist_ok=True)
		results, models = [], []
		for model_key, model in self.model_dict.items():
			print('{} is running...'.format(model_key))
			result = self.predict(model_key)
			results.append(result); models.append(model)
			explainer = SingleResultExplainer(model, self.pa_hpo_list, result, diag_list, self.TOP_N)
			explainer.explainSave(os.path.join(folder, model_key+'.txt'))
		multi_model_explainer = MultiModelExplainer(self.pa_hpo_list, models, results, diag_list, [1, 5, 10, 20])
		multi_model_explainer.explainSave(os.path.join(folder, 'AllModel.txt'))


if __name__ == '__main__':
	pass
