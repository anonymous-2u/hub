from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#!/usr/bin/env python
import datetime
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
import sys
import operator
#from classify import Classify
import os
from shutil import copyfile
from threading import Thread
import mysql.connector
class DBConnection:

    def get_connection(self):
        conn = mysql.connector.connect(host="localhost", user="root", passwd="Soumya@97")
        mycursor = conn.cursor()
        sql = "use my_db"
        mycursor.execute(sql)
        conn.commit() 
        return conn
class Classify():
    def __init__(self,path):
        self.path = path



    def load_graph(self,model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def read_tensor_from_image_file(self,file_name, input_height=299, input_width=299,input_mean=0, input_std=255):
        input_name = "file_reader"
        output_name = "normalized"
        file_reader = tf.read_file(file_name, input_name)
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(file_reader, channels = 3,
                                               name='png_reader')
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                          name='gif_reader'))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
        else:
            image_reader = tf.image.decode_jpeg(file_reader, channels = 3,name='jpeg_reader')
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0);
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        sess = tf.Session()
        result = sess.run(normalized)

        return result

    def load_labels(self,label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def classify_image(self):
        image_path = self.path
        input_height = 224
        input_width = 224
        input_mean = 128
        input_std = 128
        input_layer = "input"
        output_layer = "final_result"
        model_file = 'retrained_graph.pb'
        label_file = 'retrained_labels.txt'

        graph = self.load_graph(model_file)
        t = self.read_tensor_from_image_file(image_path,input_height=input_height,input_width=input_width,input_mean=input_mean,input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        input_operation = graph.get_operation_by_name(input_name);
        output_operation = graph.get_operation_by_name(output_name);
        with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],{input_operation.outputs[0]: t})
            end=time.time()
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        labels = self.load_labels(label_file)

        template = "{} (score={:0.5f})"
        classPred = {}
        for i in top_k:
            classPred[labels[i]] = results[i]
        ans = max(classPred, key=classPred.get)
        return ans, classPred[ans]

class Buffer:
    def add_to_buffer(self):
        count = 0
        db = DBConnection()
        conn = db.get_connection()
        mycursor = conn.cursor()
        mycursor.execute("SELECT path FROM vidbuffer")
        vid_buffer_items = mycursor.fetchall()
        for row in vid_buffer_items:
            should_continue = 1
            path = row[0]
            vidcap = cv2.VideoCapture(path)
            success,image = vidcap.read()
            while success :
                img_path = "../buffer/frame"+str(count)+".jpg"
                cv2.imwrite(img_path, image)
                db1 = DBConnection()
                conn1 = db1.get_connection()
                mycursor1 = conn1.cursor()
                mycursor1.execute("INSERT INTO `buffer`(`path`) VALUES (%s);", [img_path])
                conn1.commit()
                count = count + 1
                for _ in range(25):
                    success,image = vidcap.read()
                mycursor1.execute("SELECT continue_buffer FROM smbool where flag_var = 0")
                row1 = mycursor1.fetchone()
                should_continue = row1[0]
                print("should continue",should_continue)
                mycursor1.execute("UPDATE smbool set continue_buffer = 1 where flag_var = 0")
                conn1.commit()
                if should_continue == 0:
                    break
                mycursor1.execute("DELETE FROM `vidbuffer` WHERE `path` = %s",[path])
                conn1.commit()


class PredictAccident:
    def predict_accident(self):
        insert_into_DB = 1
        db = DBConnection()
        conn = db.get_connection()
        mycursor = conn.cursor()
        mycursor.execute("SELECT path FROM buffer")
        buffer_items = mycursor.fetchall()
        for path_row in buffer_items:
            path = path_row[0]
            clf = Classify(path)
            class_name, percentage = clf.classify_image()
            if (class_name[0] == 'a' or class_name[0] == 'A') and (insert_into_DB == 1):
                insert_into_DB = 0
                print('accident detected')
                Camera_id = 'CAM001'
                db1 = DBConnection()
                conn1 = db1.get_connection()
                mycursor1 = conn1.cursor()
                mycursor1.execute("SELECT count(path) FROM Accident")
                count_row = mycursor1.fetchone()
                new_path = '../accident/Accident'+str(count_row[0])+'.jpg'
                copyfile(path, new_path)
                date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                timestamp = time.time()
                sql1 = "insert into Accident(Camera_id,path,date_time,timestampAcc) values(%s,%s,%s,%s);"
                mycursor1.execute(sql1,[Camera_id,new_path,date_time,int(timestamp)])
                conn1.commit()
                mycursor1.execute("UPDATE flag set flag_var = 1 where flag_key = 1;")
                conn1.commit()
                mycursor1.execute("UPDATE smbool set continue_buffer = 0 where flag_var = 0")
                conn1.commit()
            if(insert_into_DB == 0):
                print('skipping database entry')
            sql = "DELETE FROM buffer WHERE path = %s"
            mycursor.execute(sql,[path])
            conn.commit()
            os.remove(path)
if __name__ == '__main__':
	while True:
		Thread(target = Buffer().add_to_buffer()).start()
		Thread(target = PredictAccident().predict_accident()).start()
            