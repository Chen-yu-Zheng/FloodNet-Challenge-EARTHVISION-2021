import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import socket
# import getpass
import json
import re
from collections import defaultdict
from nltk.stem.porter import *
import string
from nltk.tokenize import word_tokenize
from skimage.io import imread

import sys
import matplotlib.pyplot as plt

#preprocessing uses code found: https://github.com/zcyang/imageqa-san/blob/master/data_vqa/process_function.py
def process_sentence(sentence):
    periodStrip  = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip   = re.compile("(\d)(\,)(\d)")
    punct        = [';', r"/", '[', ']', '"', '{', '}',
                    '(', ')', '=', '+', '\\', '_', '-',
                    '>', '<', '@', '`', ',', '?', '!']
    contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't", \
                    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", \
                    "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd", "hed've": "he'd've", \
                    "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's", "id've": "i'd've", "i'dve": "i'd've", \
                    "im": "i'm", "ive": "i've", "isnt": "isn't", "itd": "it'd", "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", \
                    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", \
                    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't", \
                    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've", \
                    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", \
                    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll", \
                    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've", \
                    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd", "somethingd've": "something'd've", \
                    "something'dve": "something'd've", "somethingll": "something'll", "thats": "that's", "thered": "there'd", "thered've": "there'd've", \
                    "there'dve": "there'd've", "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", \
                    "they'dve": "they'd've", "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", \
                    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll", "whatre": "what're", \
                    "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd", "wheres": "where's", "whereve": "where've", \
                    "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", \
                    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've", \
                    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've", \
                    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've", "you'dve": "you'd've", \
                    "youll": "you'll", "youre": "you're", "youve": "you've"}

    inText = sentence.replace('\n', ' ')
    inText = inText.replace('\t', ' ')
    inText = inText.strip()
    
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or \
           (re.search(commaStrip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
            
    outText = periodStrip.sub("", outText, re.UNICODE)
    outText = outText.lower().split()
    for wordId, word in enumerate(outText):
        if word in contractions:            
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    
    return outText

def process_answer(answer):
    articles = ['a', 'an', 'the']
    manualMap = { 'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three':
                  '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7',
                  'eight': '8', 'nine': '9', 'ten': '10' }
    new_answer = process_sentence(answer)
    outText = []
    for word in new_answer.split():
        if word not in articles:
            word = manualMap.setdefault(word, word)
            outText.append(word)
    return ' '.join(outText)

def myimshow(image, ax=plt):
    ax.figure()
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

# custom torch Dataset
class FloodNetDataset(td.Dataset):
    def __init__(self, images_dir= './', q_dir= 'F1/Questions', mode='Training', image_size=(384, 512), top_num=40):
        super(FloodNetDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.root_image = images_dir
        self.top_num=top_num
        self.images = []
        
        if self.mode == 'Training':
            for F in os.listdir(self.root_image):
                if 'F' in F:
                    for ImageorQuesFloder in os.listdir(os.path.join(self.root_image, F)):
                        if 'Images' in ImageorQuesFloder:
                            for TorVImages in os.listdir(os.path.join(self.root_image, F, ImageorQuesFloder)):
                                if 'Train' in TorVImages:
                                    for image in os.listdir(os.path.join(self.root_image, F, ImageorQuesFloder, TorVImages)):
                                        self.images.append(os.path.join(self.root_image, F, ImageorQuesFloder, TorVImages, image))

            root_q = os.path.join(q_dir + "/%s Question.json" % mode)
            root_ans = root_q

            # print(root_q)
            with open(root_q) as f:
                self.q_json = json.load(f)
            
            with open(root_ans) as f:
                self.a_json = json.load(f)
            
            # print(self.a_json[str(4510)]['Ground_Truth'])
            # print(len(self.a_json))
            # answering parsing
                
            self.answers = []
            self.vocab_a = defaultdict(int)
            
            for i in range(len(self.a_json)):
                
                answer = str(self.a_json[str(i)]['Ground_Truth'])
                processed = process_answer(answer)
                # print(processed)
                self.answers.append(processed)
                
                if len(processed.split(" ")) == 1 or len(processed.split(" ")) == 2:
                    self.vocab_a[processed] += 1
            
            # print(self.vocab_a)
            self.vocab_a = sorted(self.vocab_a.items(), key=lambda x : x[1], reverse=True)
            # print(len(self.vocab_a))
            self.vocab_a = {self.vocab_a[i][0] : i for i in range(top_num)}
            # print(len(self.vocab_a))

            
            self.top_answers = []
            self.top_questions = []
            self.top_images = []
            
            #print(len(self.answers))


            for i, each in enumerate(self.answers):
                if all(word in self.vocab_a for word in each.split(" ")) and (len(each.split(" ")) == 1):
                    #print(each)
                    self.top_answers.append(each)
                    self.top_questions.append(process_sentence(self.q_json[str(i)]['Question']))
                    self.top_images.append(str(self.q_json[str(i)]['Image_ID']))
            
            # print(self.top_questions)
            

            # question parsing
            
            self.vocab_q = set()
            for q in self.top_questions:
                for each in q.split(" "):
                    self.vocab_q.add(each)

            # print(self.vocab_q)
            self.vocab_q = {word : i+1 for i, word in enumerate(self.vocab_q)}
            # print(self.vocab_q)
            self.vocab_q['#'] = 0 # add padding
            # print(self.vocab_q)
            
            self.seq_question = max([len(x.split(" ")) for x in self.top_questions])
            # print(self.seq_question)

        if self.mode == 'Valid':
            for F in os.listdir(self.root_image):
                if 'F' in F:
                    for ImageorQuesFloder in os.listdir(os.path.join(self.root_image, F)):
                        if 'Images' in ImageorQuesFloder:
                            for TorVImages in os.listdir(os.path.join(self.root_image, F, ImageorQuesFloder)):
                                if 'Valid' in TorVImages:
                                    for image in os.listdir(os.path.join(self.root_image, F, ImageorQuesFloder, TorVImages)):
                                        self.images.append(os.path.join(self.root_image, F, ImageorQuesFloder, TorVImages, image))
            
            root_q = os.path.join(q_dir + "/%s Question.json" % mode)
            # print(root_q)
            root_ans = os.path.join(q_dir + "/Training Question.json")

            with open(root_q) as f:
                self.q_json = json.load(f)

            with open(root_ans) as f:
                self.a_json = json.load(f)
            
            # print(self.a_json[str(4510)]['Ground_Truth'])
            # print(len(self.a_json))
            # answering parsing
                
            self.answers = []
            self.vocab_a = defaultdict(int)
            
            for i in range(len(self.a_json)):
                
                answer = str(self.a_json[str(i)]['Ground_Truth'])
                processed = process_answer(answer)
                # print(processed)
                self.answers.append(processed)
                
                if len(processed.split(" ")) == 1 or len(processed.split(" ")) == 2:
                    self.vocab_a[processed] += 1
            
            self.vocab_a = sorted(self.vocab_a.items(), key=lambda x : x[1], reverse=True)
            self.vocab_a = {self.vocab_a[i][0] : i for i in range(top_num)}
                           
            self.questions = []
            self.vocab_q = set()
            
            for i in range(len(self.q_json)):
                
                question = str(self.q_json[str(i)]['Question'])
                processed = process_sentence(question)
                # print(processed)
                self.questions.append(processed)
                
            for q in self.questions:
                for each in q.split(" "):
                    self.vocab_q.add(each)
            
            # print(self.vocab_q)
            self.vocab_q = {word : i+1 for i, word in enumerate(self.vocab_q)}
            # print(self.vocab_q)
            self.vocab_q['#'] = 0 # add padding
            # print(self.vocab_q)
            
            self.seq_question = max([len(x.split(" ")) for x in self.questions])
            # print(self.seq_question)
            

            self.top_questions = []
            self.top_images = []
            
            #print(len(self.answers))


            for i, each in enumerate(self.questions):
                self.top_questions.append(process_sentence(self.q_json[str(i)]['Question']))
                self.top_images.append(str(self.q_json[str(i)]['Image_ID']))
            
            # print(self.top_questions)
        
    
    def __len__(self):
        return len(self.top_questions)
    
    def __repr__(self):
        return "FloodNetDataset(mode={}, image_size={})" . \
                format(self.mode, self.image_size)

    def one_hot_answer(self, inp, mapping):
        return torch.Tensor([mapping[inp]])
   
    def one_hot_question(self, inp, mapping):
        vec = torch.zeros(len(inp.split(" ")))
        
        for i, word in enumerate(inp.split(" ")):
            vec[i] = mapping[word]
        
        return vec
        
                        
    def __getitem__(self, idx):
        if self.mode == 'Training':
            q = self.top_questions[idx]
            a = self.top_answers[idx]
            img_id = self.top_images[idx]
            # print(img_id)
            
            for image_name in self.images:
                if img_id in image_name:
                    img_path = image_name
                    #print(img_path)
            
            img = Image.open(img_path).convert("RGB")
            transform = tv.transforms.Compose([tv.transforms.Resize(self.image_size) , tv.transforms.ToTensor(), tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            x = transform(img)
            # xx = x.numpy().transpose(1,2,0)
            # plt.imshow(xx)
            # plt.show()
    
            one_hot_q = self.one_hot_question(q, self.vocab_q) #valid
            one_hot_ans = self.one_hot_answer(a, self.vocab_a) #train
            
            target_q = torch.zeros(self.seq_question)
            target_q[:one_hot_q.shape[0]] = one_hot_q
            
            return x, target_q, len(one_hot_q), one_hot_ans
        
        if self.mode == 'Valid':
            q = self.top_questions[idx]
            img_id = self.top_images[idx]
            print(img_id)
            
            for image_name in self.images:
                if img_id in image_name:
                    img_path = image_name
            
            img = Image.open(img_path).convert("RGB")
            transform = tv.transforms.Compose([tv.transforms.Resize(self.image_size) , tv.transforms.ToTensor(), tv.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
            x = transform(img)
            # xx = x.numpy().transpose(1,2,0)
            # plt.imshow(xx)
            # plt.show()
    
            one_hot_q = self.one_hot_question(q, self.vocab_q) #valid
            
            target_q = torch.zeros(self.seq_question)
            target_q[:one_hot_q.shape[0]] = one_hot_q
            
            return x, target_q, len(one_hot_q)

def main():
    images_dir = './'
    q_dir = 'F1/Questions'
    mode = 'Training'

    a = FloodNetDataset(
        images_dir= images_dir, q_dir = q_dir, 
        mode= mode, image_size=(384, 512), top_num=40
    )
    print('begin')
    print(a.vocab_a)

    # x = 'non flooded'
    # print(len(process_answer(x).split(' ')))


if __name__ == '__main__':
    main()


