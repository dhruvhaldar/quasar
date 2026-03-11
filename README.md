# Quasar

Quasar is a web-based, interactive data analytics platform built for SF2935 Modern Methods of Statistical Learning. It provides hands-on visualization and hyperparameter tuning for the foundational algorithms of modern machine learning.

Built with a Deep Space theme using Tailwind CSS and the daisyUI component library, Quasar offers a highly semantic, glowing, and accessible interface for exploring complex statistical datasets.

## 📚 Syllabus Mapping (SF2935)

This project strictly adheres to the course learning outcomes and content:

*   **Supervised Learning**: Implements classification methods, including support vector machines, artificial neural networks, and decision trees.
*   **Ensemble Methods**: Features advanced predictive techniques, specifically boosting and bagging.
*   **Unsupervised Learning**: Covers unlabelled data grouping with a focus on K-means clustering and nearest neighbours.
*   **Practical Application**: Focuses primarily on the practical aspects of statistical learning through computer-aided project work and dataset manipulation.
*   **Theoretical Analysis**: Applies mathematical theory to analyze and explain the properties of these statistical learning methods.

## 🚀 Deployment (Vercel)

Quasar is designed to run as a serverless machine learning engine.

1.  Fork this repository.
2.  Deploy to Vercel (Python runtime is auto-detected).
3.  Access the Data Dashboard at `https://your-quasar.vercel.app`.

## 📊 Visualizations & Artifacts

### 1. Support Vector Machines (Margin & Boundary)
Visualizes the optimal separating hyperplane in a 2D feature space. The tool highlights the specific data points that act as Support Vectors and demonstrates how adjusting the cost parameter $C$ or switching to an RBF kernel warps the decision boundary to accommodate non-linear data.

### 2. K-means Clustering
An interactive step-by-step animation of the K-means algorithm. Users can watch the centroids randomly initialize, assign points based on Euclidean distance, and iteratively recalculate their centers until the within-cluster variance converges.

### 3. Artificial Neural Networks (MLP)
A dynamic network graph showcasing the hidden layers of a Multi-Layer Perceptron. As the model trains on the backend, a Chart.js component simultaneously graphs the training loss vs. validation loss, illustrating the Bias-Variance tradeoff and the onset of overfitting.
