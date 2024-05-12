from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier as KNN
import joblib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# mtcnn检测人脸位置
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# 生成人脸512维特征向量
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # 使用InceptionResnetV1 卷积神经网络  在vggface2人脸数据集上进行训练


def collate_fn(x):  # 数据整理 获取样本
    return x[0]


def train(path):  # 训练图像路径
    dataset = datasets.ImageFolder(path)
    dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0, batch_size=1)

    print(dataset.idx_to_class)

    aligned = []
    names = []
    i = 0
    for x, y in loader:
        try:
            x_aligned, prob = mtcnn(x, return_prob=True)  # mtcnn返回5个关键点，分别是左眼，右眼，鼻子，左嘴角，右嘴角
            if x_aligned is not None:  # 有返回值
                print(f'batch {i}')
                i += 1
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])
        except Exception as e:
            print(e)

    aligned = torch.stack(aligned).to(device)
    embeddings = resnet(aligned).detach().cpu()

    a = pd.DataFrame(np.array(embeddings), index=names)
    a.reset_index(inplace=True)
    a.columns = ['label'] + [f'v{i}' for i in range(512)]

    return a


if __name__ == '__main__':

    face_data = pd.concat([
        train('faces')
    ])

    face_data.to_csv('face_feature.csv', index=False, encoding='utf8')  # 保存为csv特征文件

    # 训练KNN模型
    x = face_data.drop(columns=['label'])
    knn = KNN(n_neighbors=5)
    y = face_data['label']
    knn.fit(x.values, y)

    if os.path.exists('knn_model.pkl'):
        os.remove('knn_model.pkl')  # 删除旧的模型

    joblib.dump(knn, 'knn_model.pkl')  # 保存新的模型
    print('导出模型')
