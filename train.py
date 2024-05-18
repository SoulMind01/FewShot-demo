from model import *
from dataset import *


class ARGUMENTS:
    def __init__(
        self,
        normal_class: int,
        num_ref: int,
        num_ref_eval: int,
        lr: float,
        vector_size: int,
        weight_decay: float,
        seed: int,
        weight_init_seed: int,
        alpha: float,
        k: int,
        epochs: int,
        contamination: float,
        batch_size: int,
        biases: int,
        dataset: str,
        distance_method: str,
        evaluation_method: str = None,
        model: VGG16 = None,
    ):
        self.normal_class = normal_class
        self.num_ref = num_ref
        self.num_ref_eval = num_ref_eval
        self.lr = lr
        self.vector_size = vector_size
        self.weight_decay = weight_decay
        self.seed = seed
        self.weight_init_seed = weight_init_seed
        self.alpha = alpha
        self.k = k
        self.epochs = epochs
        self.contamination = contamination
        self.batch_size = batch_size
        self.biases = biases
        self.dataset = dataset
        self.distance_method = distance_method
        self.evaluation_method = evaluation_method
        self.model = model


def train(
    args: ARGUMENTS, small: int = 0
) -> tuple[pd.DataFrame, float, float, float, float, float]:
    """
    return (auc, f1, precision, recall, accuracy)
    """

    def create_batches(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def dist(output1, vector, label=0):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        d_norm2 = F.pairwise_distance(output1, vector).to(device)
        d_norm1 = torch.sum(torch.pow(torch.abs(output1 - vector), 1)).to(device)
        d_norm3 = torch.sum(torch.pow(torch.abs(output1 - vector), 3)).to(device)
        d_infinity = torch.max(torch.abs(output1 - vector)).to(device)
        d_cosine = F.cosine_similarity(output1, vector).to(device)
        weights = F.softmax(
            torch.Tensor(
                [
                    d_norm1 * 0.001,
                    d_norm2 * 0.04,
                    d_norm3,
                    d_infinity * 0.04,
                    d_cosine * len(output1) * 0.01,
                ]
            ),
            dim=0,
        ).to(device)
        norms = torch.Tensor([d_norm1, d_norm2, d_norm3, d_infinity, d_cosine]).to(
            device
        )
        sum = torch.dot(weights, norms)
        return sum

    class DistLoss(torch.nn.Module):
        def __init__(self, alpha, anchor, device, v=0.0, margin=0.8):
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

            # calculate the loss
            # loss = ((1 - label) * torch.pow(euclidean_distance, 2) * 0.5) + (
            #     (label)
            #     * torch.pow(
            #         torch.max(
            #             torch.Tensor([torch.tensor(0), marg - euclidean_distance])
            #         ),
            #         2,
            #     )
            #     * 0.5
            # )

            loss = (1 - label) * euclidean_distance + label * torch.max(
                torch.Tensor([torch.tensor(0), marg - euclidean_distance])
            )

            return loss + sphericity_loss

    def L2_dist(vec1, vec2):
        return F.pairwise_distance(vec1, vec2).to(device)

    def L1_dist(vec1, vec2):
        return torch.sum(torch.abs(vec1 - vec2)).to(device)

    class ContrasstiveLoss(torch.nn.Module):
        def __init__(self, alpha, anchor, device, v=0.0, margin=0.8):
            super(ContrasstiveLoss, self).__init__()
            self.margin = margin
            self.v = v
            self.alpha = alpha
            self.anchor = anchor
            self.device = device

        def forward(self, output1, vectors, label):
            # calculate the center of vectors
            center = torch.mean(torch.stack(vectors), dim=0)
            # calculate the distance between the center and all other vectors
            distances = [L2_dist(center, i) for i in vectors]
            # use the variance of the distances as the spheicity loss
            sphericity_loss = (
                torch.var(torch.stack(distances)) if len(vectors) > 1 else 0
            )

            euclidean_distance = torch.FloatTensor([0]).to(self.device)

            # get the distance between output1 and all other vectors
            for i in vectors:
                euclidean_distance += (1 - self.alpha) * (
                    L2_dist(output1, i)
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
        if args.dataset == "mnist"
        else "data/fashion/raw" if args.dataset == "fashion" else "data/cifar10/raw"
    )

    indexes = create_reference(
        args.contamination,
        args.dataset,
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
        args.dataset,
        indexes,
        args.normal_class,
        task="train",
        data_path=data_path,
        download_data=True,
    )
    val_dataset = load_dataset(
        args.dataset,
        indexes,
        args.normal_class,
        task="test",
        data_path=data_path,
        download_data=False,
    )
    indexes_ref = create_reference(
        args.contamination,
        args.dataset,
        args.normal_class,
        "train",
        data_path,
        False,
        args.num_ref_eval,
        args.seed,
    )
    ref_dataset = load_dataset(
        args.dataset,
        indexes_ref,
        args.normal_class,
        task="train",
        data_path=data_path,
        download_data=False,
    )
    model = args.model
    inner_vector_size = {
        "fashion": 2304,
        "mnist": 2304,
        "cifar10": 4096,
    }
    model.classifier = nn.Linear(
        inner_vector_size[args.dataset], args.vector_size, bias=args.biases
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
        else ContrasstiveLoss(args.alpha, anchor, device)
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

        if epoch == args.epochs - 1 and args.evaluation_method == None:
            model.eval()
            if small != 0:
                val_dataset = torch.utils.data.Subset(val_dataset, range(small))
            loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, shuffle=True
            )
            outs = {}

            ref_images = {}
            ind = list(range(0, num_ref_eval))
            np.random.shuffle(ind)

            for i in ind:
                img1, _, _, _ = ref_dataset.__getitem__(i)
                ref_images["images{}".format(i)] = model.forward(
                    img1.to(device).float()
                )
                outs["outputs{}".format(i)] = []

            means = []
            minimum_dists = []
            labels = []

            # loop through images in dataloader
            with torch.inference_mode():
                for i, data in enumerate(loader):
                    image = data[0][0]
                    label = data[2].item()

                    labels.append(label)
                    total = 0
                    mini = torch.Tensor([1e20])
                    out = model.forward(image.to(device).float())

                    # calculate the distance from the test image to each of the datapoints in the reference set
                    if args.distance_method == "multi":
                        for j in range(0, num_ref_eval):
                            distance = (1 - args.alpha) * (
                                dist(out, ref_images["images{}".format(j)])
                                / torch.sqrt(torch.Tensor([out.size()[1]])).to(device)
                                + args.alpha
                                * dist(out, anchor)
                                / torch.sqrt(torch.Tensor([out.size()[1]])).to(device)
                            )

                            outs["outputs{}".format(j)].append(distance.item())
                            total += distance.item()
                            if distance.detach().item() < mini:
                                mini = distance.item()
                    else:
                        for j in range(0, num_ref_eval):
                            distance = (1 - args.alpha) * (
                                L2_dist(out, ref_images["images{}".format(j)])
                                / torch.sqrt(torch.Tensor([out.size()[1]])).to(device)
                                + args.alpha
                                * L2_dist(out, anchor)
                                / torch.sqrt(torch.Tensor([out.size()[1]])).to(device)
                            )

                            outs["outputs{}".format(j)].append(distance.item())
                            total += distance.item()
                            if distance.detach().item() < mini:
                                mini = distance.item()

                    minimum_dists.append(mini)
                    means.append(total / len(indexes))

    # create dataframe of distances to each feature vector in the reference set for each test feature vector
    cols = ["label", "minimum_dists", "means"]
    df = pd.concat(
        [
            pd.DataFrame(labels, columns=["label"]),
            pd.DataFrame(minimum_dists, columns=["minimum_dists"]),
            pd.DataFrame(means, columns=["means"]),
        ],
        axis=1,
    )
    for i in range(0, num_ref_eval):
        if i == len(df.columns) - 3:
            break
        df = pd.concat([df, pd.DataFrame(outs["outputs{}".format(i)])], axis=1)
        cols.append("ref{}".format(i))
    df.columns = cols
    df = df.sort_values(by="minimum_dists", ascending=False).reset_index(drop=True)

    # calculate metrics
    fpr, tpr, _ = roc_curve(np.array(df["label"]), np.array(df["minimum_dists"]))
    outputs = np.array(df["minimum_dists"])
    thres = np.percentile(outputs, 10)
    outputs1 = outputs.copy()
    outputs1[outputs > thres] = 1
    outputs1[outputs <= thres] = 0
    f1 = f1_score(np.array(df["label"]), outputs1)
    fp = len(df.loc[(outputs1 == 1) & (df["label"] == 0)])
    tn = len(df.loc[(outputs1 == 0) & (df["label"] == 0)])
    fn = len(df.loc[(outputs1 == 0) & (df["label"] == 1)])
    tp = len(df.loc[(outputs1 == 1) & (df["label"] == 1)])
    spec = tn / (fp + tn) if (fp + tn) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    acc = (recall + spec) / 2
    fpr, tpr, _ = roc_curve(np.array(df["label"]), np.array(df["means"]))
    auc = metrics.auc(fpr, tpr)

    print(
        "auc: {:.4f}, f1: {:.4f}, spec: {:.4f}, recall: {:.4f}, acc: {:.4f}".format(
            auc, f1, spec, recall, acc
        )
    )
    return df, auc, f1, spec, recall, acc
