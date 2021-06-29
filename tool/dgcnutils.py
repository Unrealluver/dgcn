import torch
import numpy as np
from scipy.ndimage import zoom
import maxflow
from tool.imutils import crf_inference
import tool.imutils as imutils


def cacul_knn_matrix_(feature_map, k=10):
    batchsize, channels, h, w = feature_map.shape
    n = h * w
    # S = torch.zeros(batchsize,n,n)
    knn_matrix = torch.zeros(batchsize, n, n, device='cuda')
    for i in range(batchsize):
        # reshape feature_map: n*channel
        feature = torch.transpose(feature_map[i].reshape(channels, h * w), 0, 1)
        # ||x1-x2||^2
        x1_norm = (feature ** 2).sum(dim=1).view(-1, 1)  # n*1
        x2_norm = x1_norm.view(1, -1)  # 1*n
        dist = (x1_norm + x2_norm - 2.0 * torch.mm(feature, feature.transpose(0, 1))).abs()  # x1_norm + x2_norm : n*n
        # first method
        value, position = torch.topk(dist, k, dim=1, largest=False)  # value's shape[n, 10]
        temp = value[:, -1].unsqueeze(1).repeat(1, n)

        knn_matrix[i] = (dist <= temp).float() - torch.eye(n, n, device='cuda')

    return knn_matrix


def crf_operation(images, probs, mean_pixel=imutils.VOC12_MEAN_RGB):
    batchsize, _, h, w = probs.shape
    probs[probs < 0.0001] = 0.0001
    # unary = np.transpose(probs, [0, 2, 3, 1])

    im = images
    im = zoom(im, (1.0, 1.0, float(h) / im.shape[2], float(w) / im.shape[3]), order=1)
    im = np.transpose(im, [0, 2, 3, 1])
    im = im + mean_pixel[None, None, None, :]
    im = np.ascontiguousarray(im, dtype=np.uint8)
    result = np.zeros(probs.shape)
    for i in range(batchsize):
        result[i] = crf_inference(im[i], probs[i])

    result[result < 0.0001] = 0.0001
    result = result / np.sum(result, axis=1, keepdims=True)

    result = np.log(result)

    return result


def generate_supervision(feature, label, cues, mask, pred, knn_matrix):
    batchsize, class_num, h, w = pred.shape
    Y = torch.zeros(batchsize, class_num, h, w)
    supervision = cues.clone()

    for i in range(batchsize):
        # get the index of the non-zero class value
        label_class = torch.nonzero(label[i])
        markers_new = np.zeros((h, w))
        # class_num is 21 / 2
        markers_new.fill(class_num)
        pos = np.where(cues[i].numpy() == 1)
        # fill the correct position the class label
        markers_new[pos[1], pos[2]] = pos[0]
        markers_new_flat = markers_new.reshape(h * w)
        for c in (label_class):
            # get the exact class index
            c_c = c[0].numpy()
            # get feature of the exact one in a batch and transpose its shape
            # feature.shape[1] is the channel num
            # feature_c = feature[i].reshape(feature.shape[1], h * w).transpose(1, 0)
            # get prediction of the exact class in a batch
            pred_c = pred[i][c[0]]
            pred_c_flat = pred_c.flatten()
            # construct the maxflow Graph
            g = maxflow.Graph[float]()
            # every pixel must be a node
            nodes = g.add_nodes(h * w)
            # get the position where the cues belong to an exact class
            # ====================Foreground Class(20)====================
            pos = np.where(markers_new_flat == c_c)
            # position 0 is represent the row
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 0, 10)
                # knn matrix's shape (6, 1681, 1681)
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            # ====================Uncertain Class====================
            pos = np.where(markers_new_flat == class_num)
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], -np.log10(pred_c_flat[node_i]), -np.log10(1 - pred_c_flat[node_i]))
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)
            # ====================Background Class(1)====================
            pos = np.where((markers_new_flat != class_num) & (markers_new_flat != c_c))
            for node_i in pos[0]:
                g.add_tedge(nodes[node_i], 10, 0)
                k_neighbor = np.where(knn_matrix[i][node_i] == 1)
                for neighbor in (k_neighbor[0]):
                    g.add_edge(nodes[node_i], nodes[neighbor], 1, 1)

            flow = g.maxflow()
            node_ids = np.arange(h * w)
            label_new = g.get_grid_segments(node_ids)

            supervision[i][c[0]] = torch.from_numpy(
                np.where(pred_c > 0.7, label_new.astype(int).reshape(h, w), supervision[i][c[0]])).float()

    return supervision


def softmax(preds, min_prob):
    preds_max = torch.max(preds, dim=1, keepdim=True)
    preds_exp = torch.exp(preds - preds_max[0])
    probs = preds_exp / torch.sum(preds_exp, dim=1, keepdim=True)
    min_prob = torch.ones((probs.shape), device='cuda') * min_prob
    probs = probs + min_prob
    probs = probs / torch.sum(probs, dim=1, keepdim=True)
    return probs


def constrain_loss(probs, crf):
    probs_smooth = torch.exp(torch.from_numpy(crf)).float().cuda()
    loss = torch.mean(torch.sum(probs_smooth * torch.log(clip(probs_smooth / probs, 0.05, 20)), dim=1))
    return loss


def cal_seeding_loss(pred, label):
    pred_bg = pred[:, 0, :, :]
    labels_bg = label[:, 0, :, :].float().to('cuda')
    pred_fg = pred[:, 1:, :, :]
    labels_fg = label[:, 1:, :, :].float().to('cuda')

    count_bg = torch.sum(torch.sum(labels_bg, dim=2, keepdim=True), dim=1, keepdim=True)
    count_fg = torch.sum(torch.sum(torch.sum(labels_fg, dim=3, keepdim=True), dim=2, keepdim=True), dim=1, keepdim=True)

    sum_bg = torch.sum(torch.sum(labels_bg * torch.log(pred_bg), dim=2, keepdim=True), dim=1, keepdim=True)
    sum_fg = torch.sum(torch.sum(torch.sum(labels_fg * torch.log(pred_fg), dim=3, keepdim=True), dim=2, keepdim=True),
                       dim=1, keepdim=True)
    loss_1 = -(sum_bg / torch.max(count_bg, torch.tensor(0.0001, device='cuda'))).mean()
    loss_2 = -(sum_fg / torch.max(count_fg, torch.tensor(0.0001, device='cuda'))).mean()
    loss_balanced = loss_1 + loss_2
    return loss_balanced


def clip(x, min, max):
    x_min = x < min
    x_max = x > max
    y = torch.mul(torch.mul(x, (~x_min).float()), (~x_max).float()) + ((x_min.float()) * min) + (x_max * max).float()
    return y


class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert (np.max(pred) <= self.nclass)
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert (matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    # 如果你说的是recall和precision假设一共有10篇文章，里面4篇是你要找的。
    # 根据你某个算法，你认为其中有5篇是你要找的，但是实际上在这5篇里面，只有3篇是真正你要找的。
    # 那么你的这个算法的precision是3/5=60%，
    # 也就是，你找的这5篇，有3篇是真正对的这个算法的recall是3/4=75%，
    # 也就是，一共有用的这4篇里面，你找到了其中三篇。请自行归纳总结。
    #
    # true positive : 3
    # false positive : 2
    # false negative : 1

    #             prediction
    #           True     False
    #     True   TP       FN
    # GT
    #     False  FP       TN

    # precision = true positive / (true positive + false positive)

    # recall = true positive / (true positive + false negative)
    def precision(self):
        recall = 0.0
        for i in xrange(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall / self.nclass

    #
    def recall(self):
        accuracy = 0.0
        for i in xrange(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy / self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            if not self.M[i, i] == 0:
                jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass) / len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert (len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass:  # and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m
