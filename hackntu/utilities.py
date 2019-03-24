import numpy as np
import re

def feature_loading(tag_wordtoix, tag_ixtoword):
        
    img_feats = []
    with open('./_features.txt', 'r') as f:
        for line in f:
            tmp1 = line.strip().split(" ")
            tmp2 = [float(l) for l in tmp1]
            img_feats.append(tmp2)
     
    img_feats = np.array(img_feats)
     
    tags, probs = [], []
    with open('./_tags.txt', 'r') as f:
        for line in f:
            tmp1 = line.strip().split(", ")
            tmp2 = []
            tmp3 = []
            for s in tmp1:
                tmp = s.split(" ")
                if len(tmp) > 1:
                    tmp4 = s.split(" ")[0]
                    tmp5 = s.split(" ")[1]
                    tmp5 = tmp5[tmp5.find("(")+1:tmp5.find(")")]
                    tmp2.append(tmp4)
                    tmp3.append(float(tmp5))
            tags.append(tmp2)
            probs.append(tmp3)
    
     
    n_samples = len(tags)
    n_feats = len(tag_wordtoix)
    tag_feats = np.zeros((n_samples,n_feats))
     
    for i in range(n_samples):
        tag = tags[i]
        prob = probs[i]
        for j in range(len(tag))[::-1]:
            if tag[j] in tag_wordtoix:
                tag_feats[i,tag_wordtoix[tag[j]]] = prob[j]
                  
    tag_feats = tag_feats
    
    tags = []
    for i in range(tag_feats.shape[0]):
        tags_ = str()
        tmp1 = np.sort(tag_feats[i])[::-1]
        tmp2 = np.argsort(tag_feats[i])[::-1]
        for j in range(10):
            tags_ = tags_ + tag_ixtoword[tmp2[j]]+" ("+str(round(tmp1[j],3))+"), "
        tags.append(tags_)
    
    return img_feats, tag_feats, tags
    
def split_path(string):
    strings = re.split('/', string)
    names = strings[-1]
    names = re.split("\W", names)
    basename = names[-2]
    extname = names[-1]
    return basename, extname
