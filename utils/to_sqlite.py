import sqlite3
from sqlite3 import Error
import pickle
import torch


def insert_vector_db(img_id, image_path, img, feat_vec):
    try:
        sqliteConnection = sqlite3.connect('reid_db.db')
        cursor = sqliteConnection.cursor()
    
        Query = """
            INSERT INTO vectorkb_table (img_id, img_path, img, vector_tensor) 
            VALUES (?,?,?,?)
            """

        array = (img_id, image_path, img, feat_vec)
        Query = cursor.execute(Query, array)
        sqliteConnection.commit()

        cursor.close()
        sqliteConnection.close()
        
    except Error as err:
        print("Connection error to vector table", err)


def insert_human_db(img_id, human_id):
    try:
        sqliteConnection = sqlite3.connect('reid_db.db')
        cursor = sqliteConnection.cursor()
    
        Query = """
            INSERT INTO human_table (img_id, human_id) 
            VALUES (?,?)
            """

        array = (img_id, human_id)
        Query = cursor.execute(Query, array)
        sqliteConnection.commit()

        cursor.close()
        sqliteConnection.close()
        
    except Error as err:
        print("Connection error to human table", err)



def insert_infer_db(record):
    try:
        sqliteConnection = sqlite3.connect('reid_db.db')
        cursor = sqliteConnection.cursor()
    
        Query = """
            INSERT INTO inference_table (query_img_id, query_img, match_1_img_id, match_1_img, match_1_dist, match_2_img_id,match_2_img, match_2_dist, match_3_img_id, match_3_img, match_3_dist) 
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """

        Query = cursor.execute(Query, record)
        sqliteConnection.commit()

        cursor.close()
        sqliteConnection.close()
        
    except Error as err:
        print("Connection error to inference table", err)


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


def load_human_db():
    try:
        sqliteConnection = sqlite3.connect('reid_db.db')
        cursor = sqliteConnection.cursor()

        cursor.execute("select img_id, human_id from human_table")
        result = cursor.fetchall()
        human_dict = dict(result)

        cursor.close()
        sqliteConnection.close()
        return human_dict

    except Error as err:
        print("Connection error to Sql", err)


def convertToBinaryData(filename):
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


