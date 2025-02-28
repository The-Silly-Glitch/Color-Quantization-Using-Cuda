import torch

class KMeansClustering:
    def __init__(self, num_clusters=10, max_iters=100, tol=1e-4):
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, pixels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pixels = torch.tensor(pixels, dtype=torch.float32, device=device)
        centroids = torch.rand((self.num_clusters, 3), dtype=torch.float32, device=device) * 255
        
        for _ in range(self.max_iters):
            # Compute distances using GPU
            distances = torch.norm(pixels[:, None, :] - centroids, dim=2)
            labels = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.stack([
                pixels[labels == i].mean(dim=0) if (labels == i).sum() > 0 
                else torch.rand(3, device=device) * 255 
                for i in range(self.num_clusters)
            ])

            # Check convergence
            if torch.norm(new_centroids - centroids) < self.tol:
                break

            centroids = new_centroids

        return centroids.cpu().numpy(), labels.cpu().numpy()
