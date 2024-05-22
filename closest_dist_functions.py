import torch
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
import sys
import io


def check_and_download(dataset_name: str):
    """Check if dataset exists locally, if not download it."""
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Capture the standard output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    try:
        if dataset_name == "mnist":
            dataset = datasets.MNIST(
                root=data_dir,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
        elif dataset_name == "fashion":
            dataset = datasets.FashionMNIST(
                root=data_dir,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
        elif dataset_name == "cifar10":
            dataset = datasets.CIFAR10(
                root=data_dir,
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
    finally:
        # Restore the standard output
        sys.stdout = old_stdout

    return dataset


def get_image(dataset: str, class_type: int) -> torch.Tensor:
    """Randomly select an image index from the dataset with the given class_type"""
    dataset = check_and_download(dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for data, target in dataloader:
        if target.item() == class_type:
            return data


def get_images(dataset: str, class_type: int, num_images: int) -> list:
    """Randomly select num_images images from the dataset with the given class_type
    and return them as a tensor of shape (num_images, 1, 28, 28) and a list of labels
    """
    images = []
    labels = []
    dataset_ = check_and_download(dataset)

    dataloader = torch.utils.data.DataLoader(dataset_, batch_size=1, shuffle=True)
    for data, target in dataloader:
        if target.item() == class_type and len(images) < num_images:
            images.append(data)
            labels.append(target.item())
        if len(images) == num_images:
            break
    return images, labels


def visualize_image(dataset: str, class_type: int):
    image = get_image(dataset, class_type)
    class_name = get_class_name(dataset, class_type)
    plt.figure(figsize=(3, 3))
    plt.title(f"class {class_name} in {dataset}")
    plt.imshow(image[0][0], cmap="gray")
    plt.axis("off")
    plt.show()


def visualize_imags(dataset: str, class_type: int, num_images: int):
    images, labels = get_images(dataset, class_type, num_images)
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"class {get_class_name(dataset, class_type)} in {dataset}")
    # Arrange the images like a square grid as much as possible
    grid_size = int(np.ceil(np.sqrt(num_images)))
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)
        class_name = get_class_name(dataset, labels[i])
        plt.title(f"class {class_name}")
        plt.imshow(images[i][0][0], cmap="gray")
        plt.axis("off")
    plt.show()


def get_class_name(dataset: str, class_type: int) -> str:
    if dataset == "mnist":
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"][class_type]
    elif dataset == "fashion":
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ][class_type]
    elif dataset == "cifar10":
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ][class_type]


def aggregate_data(
    dataset_name: str, image_num_per_class: list
) -> tuple[torch.Tensor, list]:
    images = []
    labels = []
    for i, num_images in enumerate(image_num_per_class):
        class_images, class_labels = get_images(dataset_name, i, num_images)
        images.extend(class_images)
        labels.extend(class_labels)
    data = torch.cat(images, dim=0)
    return data, labels


def construct_fewshot_dataloader(
    dataset_name: str, image_num_per_class: list
) -> torch.utils.data.DataLoader:
    data_tensor, labels = aggregate_data(dataset_name, image_num_per_class)
    data_tensor = MyDataset(dataset_name, data_tensor, labels)
    dataloader_ = torch.utils.data.DataLoader(data_tensor, batch_size=1, shuffle=True)
    return dataloader_


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, data, labels):
        self.dataset_name = dataset_name
        self.dataset = data
        self.labels = labels

    def __getitem__(self, index):
        # Check the dataset name and return the correct shape
        if self.dataset_name in ["mnist", "fashion"]:
            # For MNIST and FashionMNIST, duplicate the image to have 3 channels
            return self.dataset[index].repeat(3, 1, 1), self.labels[index]
        else:
            # For CIFAR10, the images already have 3 channels
            return self.dataset[index], self.labels[index]

    def __len__(self):
        return len(self.dataset)


def visualize_dataset_(dataset: MyDataset, num_images: int):
    labels = dataset.labels
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Dataset Visualization")
    # Arrange the images like a square grid as much as possible
    grid_size = int(np.ceil(np.sqrt(num_images)))
    length = min(len(dataset.dataset), num_images)
    for i in range(length):
        plt.subplot(grid_size, grid_size, i + 1)
        class_name = get_class_name(dataset.dataset_name, labels[i])
        plt.title(class_name)
        # Convert the tensor to numpy array and transpose the dimensions
        image = dataset[i][0].numpy().transpose((1, 2, 0))
        plt.imshow(image, cmap="gray")
        plt.axis("off")
    plt.show()


def get_feature_embeddings(model, dataloader):
    embeddings, labels = [], []
    with torch.inference_mode():
        model.eval()
        for data, label in dataloader:
            labels.append(label.item())
            embeddings.append(model(data).squeeze(dim=1).numpy())
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        embeddings = np.squeeze(embeddings, axis=1)
        return embeddings, labels


def calculate_dist(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the euclidean distance between two vectors"""
    return np.linalg.norm(x - y)


def get_closest_class(
    embeddings: np.ndarray, labels: np.ndarray, query: np.ndarray
) -> int:
    """Return the class of the closest embedding to the query"""
    distances = np.array([calculate_dist(query, embedding) for embedding in embeddings])
    closest_index = np.argmin(distances)
    min_dist = distances[closest_index]
    return labels[closest_index], min_dist
