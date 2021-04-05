import torch
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from config import Config
from utils.to_sqlite import insert_vector_db, insert_human_db, insert_infer_db, load_gallery_from_db, convertToBinaryData, load_human_db

from model import make_model
from torch.backends import cudnn
import torchvision.transforms as T
from utils.metrics import cosine_similarity, euclidean_distance
import pickle

class reid_inference:
    """Reid Inference class.
    """

    def __init__(self):
        cudnn.benchmark = True
        self.Cfg = Config()
        self.model = make_model(self.Cfg, 255)
        self.model.load_param(self.Cfg.TEST_WEIGHT)
        self.model = self.model.to('cuda')
        self.transform = T.Compose([
                T.Resize(self.Cfg.INPUT_SIZE),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        print(f'Model loaded with weight from {self.Cfg.TEST_WEIGHT}')
        self.model.eval()
        print('Ready to Eval')
        print('Loading from DB...')
        self.all_img_path, self.all_gal_feat = load_gallery_from_db() #load from vectorkb_table
        self.human_dict = load_human_db()
        self._tmp_img = ""
        self._tmp_galfeat = ""
        print('Data loaded. You can start infer an image using to_gallery_feat --> query_feat --> infer')
    


    def to_gallery_feat(self, image_path, flip=True, norm=True):
        """
        Use to build gallery feat on images picked from Deep Sort.
        This is different from normal query feature extraction as this has flipped & normed feature,
        to improve the matching precision.
        """
        query_img = Image.open(image_path)
        input = torch.unsqueeze(self.transform(query_img), 0)
        input = input.to('cuda')
        with torch.no_grad():
            if flip:
                gal_feat = torch.FloatTensor(input.size(0), 2048).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(input.size(3) - 1, -1, -1).long().cuda()
                        input = input.index_select(3, inv_idx)
                    f = self.model(input)
                    gal_feat = gal_feat + f
            else:
                gal_feat = self.model(input)

            if norm:
                gal_feat = torch.nn.functional.normalize(gal_feat, dim=1, p=2)

            self._tmp_img = image_path
            self._tmp_galfeat = gal_feat
            return gal_feat



    def to_query_feat(self, image_path):
        """
        image - input image path.
        for finding query feature, no flipping and normalization is done.
        This function returns feature (1,2048) tensor.

        **considering keeping this query_feat into db as currently there is no storage of this data
        """
        query_img = Image.open(image_path)
        input = torch.unsqueeze(self.transform(query_img), 0)
        input = input.to('cuda')
        with torch.no_grad():
            query_feat = self.model(input)
        return query_feat


    def infer(self, query_feat, query_img_path, top_k= 3, to_db=True):

        dist_mat = torch.nn.functional.cosine_similarity(query_feat, self.all_gal_feat).cpu().numpy()
        indices = np.argsort(dist_mat)[::-1]

        if to_db:

            #if match found --> insert to human_table, need a human list too. make it into class
            #if no match found --> insert new identity to human_table.
            if dist_mat[indices[0]] >= self.Cfg.THRESHOLD:
                #match found
                matched_img_id = self.all_img_path[indices[0]].split('/')[-1]
                identity = self.human_dict[matched_img_id]
                print(f"Match found! Identity is {identity}")

                #insert to human_table & dict
                query_img_id = query_img_path.split('/')[-1]
                insert_human_db(query_img_id, identity)
                self.human_dict[query_img_id] = identity


            else:
                #no match found
                new_identity = str(int(max(self.human_dict.values()))+1)
                print(f"No match found! Creating new identity -- {new_identity}")

                #insert to human_table & dict
                query_img_id = query_img_path.split('/')[-1]
                insert_human_db(query_img_id, new_identity)
                self.human_dict[query_img_id] = new_identity

            #insert query image to gallery table & list
            insert_vector_db(query_img_id, query_img_path, convertToBinaryData(query_img_path), pickle.dumps(self._tmp_galfeat) )
            self.all_img_path.append(query_img_path)
            self.all_gal_feat = torch.cat([self.all_gal_feat, self._tmp_galfeat])



            record = [query_img_path.split('/')[-1],convertToBinaryData(query_img_path)]
            #query_img_id, match_1_img_id, match_1_img, match_1_dist, match_2_img_id,match_2_img, match_2_dist, match_3_img_id, match_3_img, match_3_dist
            for k in range(top_k):
                if dist_mat[indices[k]] >= self.Cfg.THRESHOLD:
                    record.append(self.all_img_path[indices[k]].split('/')[-1])
                    record.append(convertToBinaryData(self.all_img_path[indices[k]]))
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
                img = np.asarray(Image.open(self.all_img_path[indices[k]]))
                plt.title(name)
                plt.axis('off')
                plt.imshow(img)
            plt.show()







    
    # def build_all_gallery(dir_to_gal_folder = self.Cfg.GALLERY_DIR, to_db = False):
    #     """
    #     TAKE NOTEE!! TO BE MODIFIED AS WE NO LONGER NEED TO MASS UPLOAD FROM
    #     IMG FILE TO GALLERY DB.
    #     """
    #     all_gal_feat = []
    #     all_img_id = os.listdir(dir_to_gal_folder) #this is id rather than path

    #     db_feat = []
    #     db_img = []

    #     print(f'Building gallery from {dir_to_gal_folder}...')
    #     for img in all_img_id:
    #         gal_feat = to_gallery_feat(dir_to_gal_folder + "/" + img)
    #         all_gal_feat.append(gal_feat)
    #         db_feat.append(pickle.dumps(gal_feat))
    #         db_img.append(convertToBinaryData(dir_to_gal_folder + "/" + img))

    #     all_gal_feat = torch.cat(all_gal_feat, dim=0)

    #     if to_db:
    #         db_img_path = [dir_to_gal_folder + "/" + img for img in all_img_id]
    #         db_humam_id = [img.split('_')[0] for img in all_img_id]
    #         insert_vector_db(all_img_id, db_img_path, db_img, db_feat)
    #         insert_human_db(all_img_id, db_humam_id)
    #         print('All gallery uploaded to DB.')
    #     else:
    #         return all_gal_feat, all_img_id