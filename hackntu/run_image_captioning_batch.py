"""
Demo for Semantic Compositional Network https://arxiv.org/pdf/1611.08002.pdf
Developed by Zhe Gan, zhe.gan@duke.edu, March, 23, 2018
"""

from os import system
from os import listdir
from image_captioning import caption_generation
from utilities import feature_loading
import pickle
import datetime

if __name__ == '__main__': 
    
    print('start image captioning @ '+str(datetime.datetime.now().time()))
    
    mypath = "examples_images/"
    image_names = listdir(mypath)
    
    file = open("image_list.lst","w")
    for img in image_names:
        file.write(mypath+img+"\n")
    file.close()
    
    x = pickle.load(open("./pretrained_model/tag_vocab.p","rb"))
    tag_wordtoix, tag_ixtoword = x[0],x[1]
    del x    

    # This step requirs windows system
    print("First, extract image features ...")
    system('.\\bin\Release\demo.exe /m model /i image_list.lst') # feature extractor
    # img_feats and tag_feats are used for generating captions
    img_feats, tag_feats, tags = feature_loading(tag_wordtoix, tag_ixtoword) 
    
    print("Now, start image captioning ...")
    N = 6 # define how many ensembles to use
    predtext = caption_generation(N, img_feats, tag_feats)
    
    for i in range(len(image_names)):
        print("Image name: "+image_names[i])
        print("Detected tags: "+tags[i])
        print("Generated captions: "+predtext[i][0])
        print(" ")
    
    print('end @ '+str(datetime.datetime.now().time()))

    
