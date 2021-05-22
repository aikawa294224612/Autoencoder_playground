from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

digits = load_digits()
data = digits.data

pca=PCA(n_components=2)
after_pca = pca.fit(data).transform(data)

print(after_pca)
print(pca.explained_variance_ratio_)

plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10,6))
c_map = plt.cm.get_cmap('jet', 10)
plt.scatter(after_pca[:, 0], after_pca[:, 1], s = 15,
            cmap='tab10', c=digits.target)
plt.axis("off")
plt.colorbar()
plt.xlabel('PC-1'), plt.ylabel('PC-2')
plt.show()
