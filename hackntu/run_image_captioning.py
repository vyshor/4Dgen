from os import system
import urllib.request
from image_captioning import caption_generation
from utilities import feature_loading
import argparse
import pickle
import datetime

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser(description='Image Captioning ')
    parser.add_argument( "--image_url", help="the url of the input image" , default='http://www.trainyourpup.biz/cm/dpl/images/create/frisbee1.jpg')
    parser.add_argument( "--image_path", help="the url of the input image" , default='')
    args = parser.parse_args()
    
    print('start image captioning @ '+str(datetime.datetime.now().time()))
    if args.image_path == '':
        image_name = args.image_url.split('/')[-1]
    
        print("first, downloading the image ...")
        urllib.request.urlretrieve(args.image_url, "./Images/"+image_name)
    else:
        image_name = args.image_path
#    image_name = "image2.png"
    file = open("temp.lst","w")
    file.write("./"+image_name)
    file.close()
    
    x = pickle.load(open("./pretrained_model/tag_vocab.p","rb"))
    tag_wordtoix, tag_ixtoword = x[0],x[1]
    del x    

    # This step requirs windows system
    print("second, extract image features ...")
    system('.\\bin\Release\demo.exe /m model /i temp.lst') # feature extractor
    # img_feats and tag_feats are used for generating captions
    img_feats, tag_feats, tags = feature_loading(tag_wordtoix, tag_ixtoword) 
    
    print("Now, start image captioning ...")
    N = 6 # define how many ensembles to use
    predtext = caption_generation(N, img_feats, tag_feats)
    
    print("Detected tags: "+tags[0])
    print("Generated captions: "+predtext[0][0])
    with open('captions.txt', 'w+') as f:
        f.write(predtext[0][0])     
    print('end @ '+str(datetime.datetime.now().time()))

    
