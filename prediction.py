import os
import numpy as np
import shutil
import time
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision as tv
import torch
import torch.utils.data as data
from models import *
from preprocess import *
from train import *

mode = 'Valid'


# list 转成Json格式数据
def listToJson(lst):
    import json
    import numpy as np
    keys = [str(x) for x in np.arange(len(lst))]
    list_json = dict(zip(keys, lst))
    str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
    return str_json


Valid_set = FloodNetDataset(images_dir=images_dir, q_dir=q_dir, mode=mode, image_size=(384, 512), top_num=40)
print(Valid_set.__len__())

class SANPrediction(SANExperiment):
    def __init__(self, valid_set, output_dir, batch_size=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.valid_set = valid_set
        torch.backends.cudnn.benchmark = False
        self.val_loader = data.DataLoader(self.valid_set, batch_size=batch_size, collate_fn=collate_fn)
        self.image_model = ResNet(output_features=1024).to(self.device)
        self.question_model = LSTM(vocab_size=len(self.valid_set.vocab_q), embedding_dim=1000,
                                   batch_size=batch_size, hidden_dim=1024).to(self.device)
        self.attention = AttentionNet(num_classes=1000, batch_size=batch_size,
                                      input_features=1024, output_features=512).to(self.device)

        self.checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'ImageModel': self.image_model.state_dict(),
                'QuestionModel': self.question_model.state_dict(),
                'AttentionModel': self.attention.state_dict()
                }

    def load_state_dict(self, checkpoint):
        # load from pickled checkpoint
        self.image_model.load_state_dict(checkpoint['ImageModel'])
        self.question_model.load_state_dict(checkpoint['QuestionModel'])
        self.attention.load_state_dict(checkpoint['AttentionModel'])

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def Prediction(self):
        print("start load")
        self.load()

        self.image_model.eval()
        self.question_model.eval()
        self.attention.eval()

        loader = self.val_loader
        print("finish load")
        print(train_set.vocab_a)
        print(train_set.get_answer(2, train_set.vocab_a))
        file = open(os.path.join("czh", 'answer.json'), 'w')
        with torch.no_grad():


            for i, q, s in loader:
                if (self.device == 'cuda'):
                    i, q, s = i.cuda(), q.cuda(), s.cuda()

                i, q, s = Variable(i), Variable(q), Variable(s)

                # forward prop validation image/question through model
                image_embed = self.image_model(i)
                question_embed = self.question_model(q.long(), s.long())
                output = self.attention(image_embed, question_embed)

                _, y_pred = torch.max(output, 1)

                print(y_pred.item())
                print(y_pred)

                answer =train_set.get_answer(y_pred.item(), train_set.vocab_a)
                file.write(answer)
                answer_line='\n'
                file.write(answer_line)

            #answer_json = listToJson(answer)
            #print(answer_json)

        file.close()



if __name__ == '__main__':
    Predic = SANPrediction(output_dir="./zcy/lr5e-3_bs16", valid_set=Valid_set)
    #print(train_set.vocab_a.items())
    # print(train_set.top_answers)
    Predic.Prediction()
    file = open(os.path.join("czh", 'answer.json'), 'w')