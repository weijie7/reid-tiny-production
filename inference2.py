import torch
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from config import Config
import pickle
from utils.to_sqlite import *
Cfg = Config()

from model import make_model
from torch.backends import cudnn
import torchvision.transforms as T
from utils.metrics import cosine_similarity, euclidean_distance
from scipy.spatial.distance import cosine

# load model and eval it
def load_model():
    global device
    global model
    global transform
    cudnn.benchmark = True
    model = make_model(Cfg, 255)
    model.load_param(Cfg.TEST_WEIGHT)
    device = 'cuda'
    model = model.to(device)
    transform = T.Compose([
        T.Resize(Cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    print(f'Model loaded with weight from {Cfg.TEST_WEIGHT}')
    model.eval()
    print('Ready to Eval')



def to_gallery_feat(image_path, flip=True, norm=True):
    """
    image - input image path.
    for building gallery feature, flip and norm should be turned on for better mAP.
    This function returns flipped and normalized feature (1,2048) tensor.
    """
    query_img = Image.open(image_path)
    input = torch.unsqueeze(transform(query_img), 0)
    input = input.to(device)
    with torch.no_grad():
        if flip:
            gal_feat = torch.FloatTensor(input.size(0), 2048).zero_().cuda()
            for i in range(2):
                if i == 1:
                    inv_idx = torch.arange(input.size(3) - 1, -1, -1).long().cuda()
                    input = input.index_select(3, inv_idx)
                f = model(input)
                gal_feat = gal_feat + f
        else:
            gal_feat = model(input)

        if norm:
            gal_feat = torch.nn.functional.normalize(gal_feat, dim=1, p=2)

    return gal_feat




def convertToBinaryData(filename):
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def build_all_gallery(dir_to_gal_folder = Cfg.GALLERY_DIR, to_db = False):
    """
    Return all_gal_feat and image path. To move completely to DB
    """
    all_gal_feat = []
    all_img_id = os.listdir(dir_to_gal_folder) #this is id rather than path

    db_feat = []
    db_img = []

    print(f'Building gallery from {dir_to_gal_folder}...')
    for img in all_img_id:
        gal_feat = to_gallery_feat(dir_to_gal_folder + "/" + img)
        all_gal_feat.append(gal_feat)
        db_feat.append(pickle.dumps(gal_feat))
        db_img.append(convertToBinaryData(dir_to_gal_folder + "/" + img))

    all_gal_feat = torch.cat(all_gal_feat, dim=0)

    if to_db:
        db_img_path = [dir_to_gal_folder + "/" + img for img in all_img_id]
        db_humam_id = [img.split('_')[0] for img in all_img_id]
        insert_vector_db(all_img_id, db_img_path, db_img, db_feat)
        insert_human_db(all_img_id, db_humam_id)
        print('All gallery uploaded to DB.')
    else:
        return all_gal_feat, all_img_id


def unpickle_blob(blob):
    if type(blob) is list:
        return torch.cat(list((pickle.loads(bob) for bob in blob)), dim=0)
        
    else:
        return pickle.loads(blob)


def load_gallery_from_db():
    try:
        sqliteConnection = sqlite3.connect('reid_db.db')
        cursor = sqliteConnection.cursor()

        cursor.execute("select img_path, vector_tensor from vectorkb_table")
        result = cursor.fetchall()
        img_path =list(list(zip(*result))[0])
        gal_feat = unpickle_blob(list(list(zip(*result))[1]))

        cursor.close()
        sqliteConnection.close()
        return img_path, gal_feat

    except Error as err:
        print("Connection error to Sql", err)




def to_query_feat(image_path):
    """
    image - input image path.
    for finding query feature, no flipping and normalization is done.
    This function returns feature (1,2048) tensor.
    """
    query_img = Image.open(image_path)
    input = torch.unsqueeze(transform(query_img), 0)
    input = input.to(device)
    with torch.no_grad():
        query_feat = model(input)

    return query_feat



def infer(query_feat, query_img_path, all_img_path, all_gal_feat, thres=0.55, top_k= 3, to_db=False):

    dist_mat = torch.nn.functional.cosine_similarity(query_feat, all_gal_feat).cpu().numpy()
    indices = np.argsort(dist_mat)[::-1]

    if to_db:

        #if match found --> insert to human_table, need a human list too. make it into class
        #if no match found --> insert new identity to human_table.
        if dist_mat[indices[0]] >= thres:
            print("Match found!")

        record = [query_img_path.split('/')[-1],convertToBinaryData(query_img_path)]
        #query_img_id, match_1_img_id, match_1_img, match_1_dist, match_2_img_id,match_2_img, match_2_dist, match_3_img_id, match_3_img, match_3_dist
        for k in range(top_k):
            if dist_mat[indices[k]] >= thres:
                record.append(all_img_path[indices[k]].split('/')[-1])
                record.append(convertToBinaryData(all_img_path[indices[k]]))
                record.append(dist_mat.item(indices[k]))
            else:
                record.append(None)
                record.append(None)
                record.append(None)
        insert_infer_db(record)
            
    else:
        plt.subplot(1, top_k+2, 1)
        plt.title('Query')
        plt.axis('off')
        query_img = Image.open(query_img_path)
        plt.imshow(np.asarray(query_img))

        for k in range(top_k):
            plt.subplot(1, top_k+2, k+3)
            name = str(indices[k]) + '\n' + '{0:.2f}'.format(dist_mat[indices[k]])
            img = np.asarray(Image.open(all_img_path[indices[k]]))
            plt.title(name)
            plt.axis('off')
            plt.imshow(img)
        plt.show()




if __name__ == "__main__":
    load_model()
    build_all_gallery(to_db=True)