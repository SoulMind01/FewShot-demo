import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data


def get_image(dataset: str, class_type: int) -> torch.Tensor:
    """randomly select an image index from the dataset with the given class_type"""
    dataset = (
        datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )
        if dataset == "mnist"
        else (
            datasets.FashionMNIST(
                root="./data",
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
            if dataset == "fashion"
            else datasets.CIFAR10(
                root="./data",
                train=False,
                download=True,
                transform=transforms.ToTensor(),
            )
        )
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    for data, target in dataloader:
        if target.item() == class_type:
            return data


def get_images(dataset: str, class_type: int, num_images: int) -> list:
    """randomly select num_images images from the dataset with the given class_type
    and return them as a tensor of shape (num_images, 1, 28, 28) and a list of labels
    """
    images = []
    labels = []
    if dataset == "cifar10":
        dataset_ = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )
    elif dataset == "mnist":
        dataset_ = datasets.MNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )
    else:
        dataset_ = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transforms.ToTensor()
        )
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
    # arange the images like a square grid as much as possible
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
        # duplicate the image to have 3 channels
        return self.dataset[index].repeat(3, 1, 1), self.labels[index]

    def __len__(self):
        return len(self.dataset)


def visualize_dataset_(dataset: MyDataset, num_images: int):
    labels = dataset.labels
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Dataset Visualization")
    # arange the images like a square grid as much as possible
    grid_size = int(np.ceil(np.sqrt(num_images)))
    length = min(len(dataset.dataset), num_images)
    for i in range(length):
        plt.subplot(grid_size, grid_size, i + 1)
        class_name = get_class_name(dataset.dataset_name, labels[i])
        plt.title(class_name)
        # convert the tensor to numpy array and transpose the dimensions
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
    """calculate the euclidean distance between two vectors"""
    return np.linalg.norm(x - y)


def get_closest_class(
    embeddings: np.ndarray, labels: np.ndarray, query: np.ndarray
) -> int:
    """return the class of the closest embedding to the query"""
    distances = np.array([calculate_dist(query, embedding) for embedding in embeddings])
    closest_index = np.argmin(distances)
    min_dist = distances[closest_index]
    return labels[closest_index], min_dist
