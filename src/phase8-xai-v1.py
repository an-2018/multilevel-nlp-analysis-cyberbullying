# phase8-xai-analysis.py
import sys
from collections import defaultdict

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import shap
import lime
import lime.lime_tabular
import lime.lime_text
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import load_model
from tqdm import tqdm
import tensorflow as tf

# Configure SHAP to handle TensorFlow models
shap.initjs()

class XAIAnalyzer:
    def __init__(self, model_paths, data_path):
        self.models = self._load_models(model_paths)
        self.data = pd.read_csv(data_path)
        self.feature_groups = self._define_feature_groups()
        self.explanations = {}

    def _load_models(self, model_paths):
        """Load trained models from different phases"""
        models = {}
        for name, path in model_paths.items():
            if path.suffix == '.pkl':
                models[name] = joblib.load(path)
            elif path.suffix == '.h5':
                models[name] = load_model(path)
        return models

    def _define_feature_groups(self):
        """Map features to their categories"""
        return {
            'lexical': ['word_count', 'unique_word_ratio', 'profanity_score'],
            'syntactic': [col for col in self.data if col.startswith('pos_')],
            'semantic': ['semantic_vector'],
            'sentiment': [col for col in self.data if col.startswith('sentiment_')],
            'tfidf': [col for col in self.data if col.startswith('tfidf_')]
        }

    def _preprocess_data(self, model_name):
        """Prepare data for specific model"""
        if 'svm' in model_name:
            return self.data[self.feature_groups['tfidf']].values
        elif 'lstm' in model_name:
            return np.stack(self.data['semantic_vector'].apply(np.array))
        return self.data.drop(columns=['label', 'clean_text']).values

    def shap_analysis(self, model_name, sample_size=100):
        """Perform SHAP analysis on selected model"""
        print(f"\nPerforming SHAP analysis for {model_name}")
        model = self.models[model_name]
        data = self._preprocess_data(model_name)
        sample_idx = np.random.choice(data.shape[0], sample_size, replace=False)
        background = shap.sample(data, 50) if data.shape[0] > 50 else data

        if isinstance(model, Pipeline) or hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            explainer = shap.DeepExplainer(model, background)

        shap_values = explainer.shap_values(data[sample_idx])

        # Visualization
        plt.figure()
        shap.summary_plot(shap_values, data[sample_idx], feature_names=self.data.columns)
        plt.savefig(f'results/phase8/shap_{model_name}.png')
        plt.close()

        self.explanations[model_name] = {'shap': shap_values}

    def lime_analysis(self, model_name, sample_size=20):
        """Perform LIME analysis on selected model"""
        print(f"\nPerforming LIME analysis for {model_name}")
        model = self.models[model_name]
        data = self._preprocess_data(model_name)

        # Create explainer based on data type
        if 'text' in model_name:
            explainer = lime.lime_text.LimeTextExplainer(class_names=['non-harmful', 'harmful'])
            samples = self.data['clean_text'].sample(sample_size).tolist()
        else:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=data,
                feature_names=self.data.columns,
                class_names=['non-harmful', 'harmful'],
                mode='classification'
            )
            samples = data[np.random.choice(data.shape[0], sample_size)]

        explanations = []
        for i in tqdm(range(sample_size)):
            exp = explainer.explain_instance(
                samples[i],
                model.predict_proba,
                num_features=10,
                top_labels=1
            )
            explanations.append(exp)

            # Save example visualization
            if i < 3:
                plt.figure()
                exp.as_pyplot_figure()
                plt.savefig(f'results/phase8/lime_{model_name}_example{i}.png')
                plt.close()

        self.explanations[model_name]['lime'] = explanations

    def integrated_gradients_analysis(self, model_name, sample_size=50):
        """Perform Integrated Gradients analysis for neural models"""
        print(f"\nPerforming Integrated Gradients analysis for {model_name}")
        model = self.models[model_name]
        data = self._preprocess_data(model_name)

        # Convert to TensorFlow dataset
        baseline = np.zeros_like(data[:sample_size])
        inputs = tf.convert_to_tensor(data[:sample_size])

        # Calculate integrated gradients
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions = model(inputs)

        grads = tape.gradient(predictions, inputs)
        ig = (inputs - baseline) * grads.numpy()

        # Aggregate and visualize
        attributions = np.mean(np.abs(ig), axis=0)
        top_features = np.argsort(attributions)[-10:]

        plt.figure()
        plt.bh(range(len(top_features)), attributions[top_features])
        plt.yticks(range(len(top_features)), [self.data.columns[i] for i in top_features])
        plt.title('Integrated Gradients Feature Importance')
        plt.savefig(f'results/phase8/ig_{model_name}.png')
        plt.close()

        self.explanations[model_name]['ig'] = ig

    def analyze_feature_categories(self):
        """Analyze feature category impacts across models"""
        category_impacts = {}
        for model_name, explanations in self.explanations.items():
            if 'shap' in explanations:
                shap_values = np.mean(np.abs(explanations['shap']), axis=0)
                category_impacts[model_name] = self._aggregate_by_category(shap_values)

        # Visualization
        pd.DataFrame(category_impacts).plot(kind='bar', stacked=True)
        plt.title('Feature Category Impacts Across Models')
        plt.ylabel('Mean Absolute SHAP Value')
        plt.savefig('results/phase8/category_impacts.png')
        plt.close()

    def _aggregate_by_category(self, values):
        """Aggregate feature importance by predefined categories"""
        impacts = {}
        for category, features in self.feature_groups.items():
            indices = [i for i, col in enumerate(self.data.columns) if col in features]
            impacts[category] = np.sum(values[indices])
        return impacts

    def generate_report(self):
        """Generate comprehensive XAI report"""
        report = []


        for model_name, explanations in self.explanations.items():
            model_report = {}
            model_report['model'] = model_name
            model_report['top_features'] = {}
            if 'shap' in explanations:
                shap_values = np.mean(np.abs(explanations['shap']), axis=0)
                top_features = np.argsort(shap_values)[-10:]
                model_report['top_features'] = {self.data.columns[i]: shap_values[i] for i in top_features}
                model_report['category_impacts'] = self._aggregate_by_category(shap_values)
            else:
                model_report['top_features'] = {self.data.columns[i]: 0 for i in range(len(self.data.columns))}
                model_report['category_impacts'] = self._aggregate_by_category(np.zeros(len(self.data.columns)))
            if 'ig' in explanations:
                model_report['ig_top_features'] = {self.data.columns[i]: np.mean(np.abs(explanations['ig'][:, i])) for i in range(len(self.data.columns))}

            if 'lime' in explanations:
                model_report['lime_top_features'] = self._process_lime_explanations(explanations['lime'])

            report.append(model_report)

            # Save report
            pd.DataFrame(report).to_csv('results/phase8/xai_report.csv', index=False)
            self._save_insights(report)

    def _process_lime_explanations(self, lime_exps):
        """Extract top features from LIME explanations"""
        feature_scores = defaultdict(float)
        for exp in lime_exps:
            for feature, score in exp.as_list():
                feature_scores[feature.split('=')[0]] += abs(score)
        return dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:10])

    def _save_insights(self, report):
        """Document key insights from XAI analysis"""
        insights = []
        for model in report:
            insights.append(f"Model: {model['model']}")
            insights.append("Top Contributing Features:")
            insights.extend([f"- {feat}: {score:.4f}" for feat, score in model['top_features'].items()])
            insights.append("\nFeature Category Impacts:")
            insights.extend([f"- {cat}: {impact:.4f}" for cat, impact in model['category_impacts'].items()])
            insights.append("\n" + "="*50 + "\n")

        with open('results/phase8/key_insights.txt', 'w') as f:
            f.write("\n".join(insights))

if __name__ == "__main__":
    # Path configuration
    data_path = Path('data/processed/phase3_output.csv')
    model_paths = {
        'tfidf_svm': Path('models/tfidf_svm.pkl'),
        'semantic_lstm': Path('models/semantic_lstm.h5'),
        'fusion_model': Path('models/fusion_model.h5')
    }

    # Create output directory
    Path('results/phase8').mkdir(parents=True, exist_ok=True)

    # Initialize and run XAI analysis
    analyzer = XAIAnalyzer(model_paths, data_path)

    # Perform analyses
    analyzer.shap_analysis('tfidf_svm')
    analyzer.lime_analysis('semantic_lstm')
    analyzer.integrated_gradients_analysis('fusion_model')

    # Generate comprehensive insights
    analyzer.analyze_feature_categories()
    analyzer.generate_report()

    print("XAI analysis complete. Results saved to results/phase8/ directory.")