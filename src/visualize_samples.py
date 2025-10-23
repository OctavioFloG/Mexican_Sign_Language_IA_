import matplotlib.pyplot as plt

def visualize_samples(X, y, categories, n=5):
    plt.figure(figsize=(12,4))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(X[i])
        plt.title(list(categories.values())[int(y[i].argmax())])
        plt.axis("off")
    plt.show()
