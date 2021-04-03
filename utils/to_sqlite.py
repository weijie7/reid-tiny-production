import sqlite3
from sqlite3 import Error


def insert_vector_db(img_id, image_path, img, feat_vec):
    try:
        sqliteConnection = sqlite3.connect('reid_db.db')
        cursor = sqliteConnection.cursor()
    
        Query = """
            INSERT INTO vectorkb_table (img_id, img_path, img, vector_tensor) 
            VALUES (?,?,?,?)
            """

        multi_record = list(zip(img_id, image_path, img, feat_vec))
        Query = cursor.executemany(Query, multi_record)
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

        multi_record = list(zip(img_id, human_id))
        Query = cursor.executemany(Query, multi_record)
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