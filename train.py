from model import *
from dataset import *
from distant_metrics import L1_dist, L2_dist, L3_dist, L4_dist, L_inf, dist
from arguments import ARGUMENTS
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


from evaluate import (
    make_predictions_by_anormaly_score,
    make_predictions_by_closest_dist,
)


def train(
    args: ARGUMENTS, small: int = 0
) -> tuple[pd.DataFrame, float, float, float, float, float, VGG16]:
    """
    Train the model and make predictions
    Args:
        args: ARGUMENTS object
        small: int - number of samples to use for evaluation
    Returns:
        tuple[pd.DataFrame, float, float, float, float, float, VGG16]: The predictions, the AUC, the F1 score, the specificity, the recall, the accuracy, and the model.
    """
    assert args.evaluation_method in [
        "anomaly_score",
        "closest_dist",
    ], "evaluation_method must be either 'anomaly_score' or 'closest_dist'"

    def create_batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    class DistLoss(torch.nn.Module):
        def __init__(
            self,
            alpha: float,
            anchor: torch.Tensor,
            device: str,
            v: float = 0.0,
            margin: float = 0.8,
        ):
            super(DistLoss, self).__init__()
            self.margin = margin
            self.v = v
            self.alpha = alpha
            self.anchor = anchor
            self.device = device

        def forward(self, output1, vectors, label):
            # calculate the center of vectors
            center = torch.mean(torch.stack(vectors), dim=0)
            # calculate the distance between the center and all other vectors
            distances = [dist(center, i) for i in vectors]
            # use the variance of the distances as the spheicity loss
            sphericity_loss = (
                torch.var(torch.stack(distances)) if len(vectors) > 1 else 0
            )

            euclidean_distance = torch.FloatTensor([0]).to(self.device)

            # get the distance between output1 and all other vectors
            for i in vectors:
                euclidean_distance += (1 - self.alpha) * (
                    dist(output1, i)
                    / torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device)
                )

            euclidean_distance += self.alpha * (
                (F.pairwise_distance(output1, self.anchor))
                / torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device)
            )

            # calculate the margin
            marg = (len(vectors) + self.alpha) * self.margin

            # if v > 0.0, apply soft-boundary
            if self.v > 0.0:
                euclidean_distance = (1 / self.v) * euclidean_distance

            loss = (1 - label) * euclidean_distance + label * torch.max(
                torch.Tensor([torch.tensor(0), marg - euclidean_distance])
            )

            return loss + sphericity_loss

    class ContrasstiveLoss(torch.nn.Module):
        def __init__(
            self,
            alpha: float,
            anchor: torch.Tensor,
            device: str,
            v: float = 0.0,
            margin=0.8,
            distance_method: str = "L2",
        ):

            super(ContrasstiveLoss, self).__init__()
            self.margin = margin
            self.v = v
            self.alpha = alpha
            self.anchor = anchor
            self.device = device

            distant_metrics = {
                "L1": L1_dist,
                "L2": L2_dist,
                "L3": L3_dist,
                "L4": L4_dist,
                "inf": L_inf,
            }
            self.distant_metric = distant_metrics[distance_method]

        def forward(self, output1, vectors, label):
            # calculate the center of vectors
            center = torch.mean(torch.stack(vectors), dim=0)
            # calculate the distance between the center and all other vectors
            distances = [self.distant_metric(center, i) for i in vectors]
            # use the variance of the distances as the spheicity loss
            sphericity_loss = (
                torch.var(torch.stack(distances)) if len(vectors) > 1 else 0
            )

            euclidean_distance = torch.FloatTensor([0]).to(self.device)

            # get the distance between output1 and all other vectors
            for i in vectors:
                euclidean_distance += (1 - self.alpha) * (
                    self.distant_metric(output1, i)
                    / torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device)
                )

            euclidean_distance += self.alpha * (
                (F.pairwise_distance(output1, self.anchor))
                / torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device)
            )

            # calculate the margin
            marg = (len(vectors) + self.alpha) * self.margin

            # if v > 0.0, apply soft-boundary
            if self.v > 0.0:
                euclidean_distance = (1 / self.v) * euclidean_distance

            loss = (1 - label) * euclidean_distance + label * torch.max(
                torch.Tensor([torch.tensor(0), marg - euclidean_distance])
            )

            return loss + sphericity_loss

    # return the feature embedding
    def init_feat_vec(model, base_ind, train_dataset, device):
        """
        Initialise the anchor
        Args:
            model object
            base_ind - index of training data to convert to the anchor
            train_dataset - train dataset object
            device
        """
        model.eval()
        anchor, _, _, _ = train_dataset.__getitem__(base_ind)
        with torch.no_grad():
            anchor = model(anchor.to(device).float())
        return anchor

    def init_anchor_average(model, train_dataset, device):
        """
        Initialise the anchor as the average of the training data
        Args:
            model object
            train_dataset - train dataset object
            device
        """
        model.eval()
        anchor = torch.zeros(args.vector_size).to(device)
        for i in range(len(train_dataset)):
            img, _, _, _ = train_dataset.__getitem__(i)
            with torch.no_grad():
                img = model(img.to(device).float())
            anchor = anchor + img
        anchor = anchor / len(train_dataset)
        return anchor

    N = args.num_ref
    num_ref_eval = args.num_ref_eval if args.num_ref_eval != None else N

    data_path = (
        "data/mnist/raw"
        if args.dataset_name == "mnist"
        else (
            "data/fashion/raw" if args.dataset_name == "fashion" else "data/cifar10/raw"
        )
    )

    indexes = create_reference(
        args.contamination,
        args.dataset_name,
        args.normal_class,
        "train",
        data_path,
        True,
        N,
        args.seed,
    )

    torch.manual_seed(args.weight_init_seed)
    torch.cuda.manual_seed(args.weight_init_seed)
    torch.cuda.manual_seed_all(args.weight_init_seed)

    train_dataset = load_dataset(
        args.dataset_name,
        indexes,
        args.normal_class,
        task="train",
        data_path=data_path,
        download_data=True,
    )
    val_dataset = load_dataset(
        args.dataset_name,
        indexes,
        args.normal_class,
        task="test",
        data_path=data_path,
        download_data=False,
    )
    indexes_ref = create_reference(
        args.contamination,
        args.dataset_name,
        args.normal_class,
        "train",
        data_path,
        False,
        args.num_ref_eval,
        args.seed,
    )
    ref_dataset = load_dataset(
        args.dataset_name,
        indexes_ref,
        args.normal_class,
        task="train",
        data_path=data_path,
        download_data=False,
    )
    model = VGG16(
        vector_size=args.vector_size,
        biases=args.biases,
        dataset_name=args.dataset_name,
        activation_function=args.activation_function,
    )

    # put the model to the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # init the anchor, optimizer, loss fn
    ind = list(range(0, len(indexes)))

    # select datapoint from ref set as anchor
    np.random.seed(args.epochs)
    rand_freeze = np.random.randint(len(indexes))
    base_ind = ind[rand_freeze]
    anchor = init_anchor_average(model, train_dataset, device)
    loss_fn = (
        DistLoss(args.alpha, anchor, device)
        if args.distance_method == "multi"
        else ContrasstiveLoss(
            args.alpha, anchor, device, distance_method=args.distance_method
        )
    )
    optimizer = (
        optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.distance_method == "multi"
        else optim.SGD(model.parameters(), lr=20, weight_decay=args.weight_decay)
    )

    train_losses = []

    for epoch in range(args.epochs):
        # print("Starting epoch " + str(epoch + 1))
        model.train()
        loss_sum = 0

        # create batches for epoch
        np.random.seed(epoch)
        np.random.shuffle(ind)
        batches = list(create_batches(ind, args.batch_size))

        for i in range(int(np.ceil(len(ind) / args.batch_size))):
            for batch_ind, index in enumerate(batches[i]):
                seed = (epoch + 1) * (i + 1) * (batch_ind + 1)
                img1, img2, labels, base = train_dataset.__getitem__(
                    index, seed, base_ind
                )

                # Forward
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)

                output1 = model.forward(img1.float())

                if args.k == 1:
                    output2 = model.forward(img2.float())
                    vecs = [output2]

                else:
                    vecs = []
                    ind2 = ind.copy()
                    np.random.seed(seed)
                    np.random.shuffle(ind2)
                    for j in range(args.k):
                        if index != ind2[j]:
                            output2 = model(
                                train_dataset.__getitem__(ind2[j], seed, base_ind)[0]
                                .to(device)
                                .float()
                            )
                            vecs.append(output2)

                if batch_ind == 0:
                    loss = loss_fn(output1, vecs, labels)
                else:
                    loss = loss + loss_fn(output1, vecs, labels)
            loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        train_losses.append((loss_sum / len(ind)))
        # print("Epoch: {}, Train loss: {}".format(epoch + 1, train_losses[-1]))

    if small != 0:
        val_dataset = torch.utils.data.Subset(val_dataset, range(small))
    if args.evaluation_method == "anomaly_score":
        return make_predictions_by_anormaly_score(
            args=args,
            model=model,
            ref_dataset=ref_dataset,
            val_dataset=val_dataset,
            anchor=anchor,
        )
    else:
        return make_predictions_by_closest_dist(
            args=args,
            model=model,
            class_size=small // (args.num_ref_eval / (args.num_ref_eval + args.k) * 10),
            test_ratio=args.num_ref_eval / (args.num_ref_eval + args.k),
        )
    #         model.eval()
    #
    #         loader = torch.utils.data.DataLoader(
    #             val_dataset, batch_size=1, shuffle=True
    #         )
    #         outs = {}

    #         ref_images = {}
    #         ind = list(range(0, num_ref_eval))
    #         np.random.shuffle(ind)

    #         for i in ind:
    #             img1, _, _, _ = ref_dataset.__getitem__(i)
    #             ref_images["images{}".format(i)] = model.forward(
    #                 img1.to(device).float()
    #             )
    #             outs["outputs{}".format(i)] = []

    #         means = []
    #         minimum_dists = []
    #         labels = []

    #         # loop through images in dataloader
    #         with torch.inference_mode():
    #             for i, data in enumerate(loader):
    #                 image = data[0][0]
    #                 label = data[2].item()

    #                 labels.append(label)
    #                 total = 0
    #                 mini = torch.Tensor([1e20])
    #                 out = model.forward(image.to(device).float())

    #                 # calculate the distance from the test image to each of the datapoints in the reference set
    #                 if args.distance_method == "multi":
    #                     for j in range(0, num_ref_eval):
    #                         distance = (
    #                             (1 - args.alpha)
    #                             * dist(out, ref_images["images{}".format(j)])
    #                             / torch.sqrt(torch.Tensor([out.size()[1]])).to(device)
    #                         )
    #                         +args.alpha * dist(out, anchor) / torch.sqrt(
    #                             torch.Tensor([out.size()[1]])
    #                         ).to(device)

    #                         outs["outputs{}".format(j)].append(distance.item())
    #                         total += distance.item()
    #                         if distance.detach().item() < mini:
    #                             mini = distance.item()
    #                 else:
    #                     for j in range(0, num_ref_eval):
    #                         distance = (1 - args.alpha) * (
    #                             L2_dist(out, ref_images["images{}".format(j)])
    #                             / torch.sqrt(torch.Tensor([out.size()[1]])).to(device)
    #                             + args.alpha
    #                             * L2_dist(out, anchor)
    #                             / torch.sqrt(torch.Tensor([out.size()[1]])).to(device)
    #                         )

    #                         outs["outputs{}".format(j)].append(distance.item())
    #                         total += distance.item()
    #                         if distance.detach().item() < mini:
    #                             mini = distance.item()

    #                 minimum_dists.append(mini)
    #                 means.append(total / len(indexes))

    # # create dataframe of distances to each feature vector in the reference set for each test feature vector
    # cols = ["label", "minimum_dists", "means"]
    # df = pd.concat(
    #     [
    #         pd.DataFrame(labels, columns=["label"]),
    #         pd.DataFrame(minimum_dists, columns=["minimum_dists"]),
    #         pd.DataFrame(means, columns=["means"]),
    #     ],
    #     axis=1,
    # )
    # for i in range(0, num_ref_eval):
    #     df = pd.concat([df, pd.DataFrame(outs["outputs{}".format(i)])], axis=1)
    #     cols.append("ref{}".format(i))
    # df.columns = cols
    # df = df.sort_values(by="minimum_dists", ascending=False).reset_index(drop=True)

    # # calculate metrics
    # fpr, tpr, _ = roc_curve(np.array(df["label"]), np.array(df["minimum_dists"]))
    # outputs = np.array(df["minimum_dists"])
    # thres = np.percentile(outputs, 10)
    # outputs1 = outputs.copy()
    # outputs1[outputs > thres] = 1
    # outputs1[outputs <= thres] = 0
    # f1 = f1_score(np.array(df["label"]), outputs1)
    # fp = len(df.loc[(outputs1 == 1) & (df["label"] == 0)])
    # tn = len(df.loc[(outputs1 == 0) & (df["label"] == 0)])
    # fn = len(df.loc[(outputs1 == 0) & (df["label"] == 1)])
    # tp = len(df.loc[(outputs1 == 1) & (df["label"] == 1)])
    # spec = tn / (fp + tn) if (fp + tn) != 0 else 0
    # recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    # acc = (recall + spec) / 2
    # fpr, tpr, _ = roc_curve(np.array(df["label"]), np.array(df["means"]))
    # auc = metrics.auc(fpr, tpr)

    # print(
    #     "auc: {:.4f}, f1: {:.4f}, spec: {:.4f}, recall: {:.4f}, acc: {:.4f}".format(
    #         auc, f1, spec, recall, acc
    #     )
    # )
    # return df, auc, f1, spec, recall, acc, model
