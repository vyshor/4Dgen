import datetime
import pickle
import numpy as np
from collections import OrderedDict

def load_params(path, param_list):

    #print('loading pretrained model...')
    
    params_set = []
    
    for num in param_list:
        params = OrderedDict()
        data = np.load('%s%s.npz'%(path, num))  
        for kk, pp in data.iteritems():
            params[kk] = data[kk].astype('float64')
        params_set.append(params)
    
    return params_set

def _p(pp, name):
    return '%s_%s' % (pp, name)

def predict(z, params_set, beam_size, max_step, prefix='encoder_lstm'):

    """ z: size of (n_z, 1)
    """
    def _slice(_x, n, dim):
        return _x[n*dim:(n+1)*dim]

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    n_h = params_set[0][_p(prefix,'Ua_i')].shape[0]

    def _step_set(x_prev, h_prev, c_prev, x_prev_id, h_prev_id, params):
        if x_prev_id >= 0 and x_prev_id in params[_p(prefix, 'cacheX')]:
            tmp1_i = params[_p(prefix, 'cacheX')][x_prev_id][0]
            tmp1_f = params[_p(prefix, 'cacheX')][x_prev_id][1]
            tmp1_o = params[_p(prefix, 'cacheX')][x_prev_id][2]
            tmp1_c = params[_p(prefix, 'cacheX')][x_prev_id][3]
        else:
            tmp1_i = np.dot((np.dot(x_prev, params[_p(prefix, 'Wa_i')]) * params[_p(prefix, 'yWb_i')]), params[_p(prefix, 'Wc_i')].T)
            tmp1_f = np.dot((np.dot(x_prev, params[_p(prefix, 'Wa_f')]) * params[_p(prefix, 'yWb_f')]), params[_p(prefix, 'Wc_f')].T)
            tmp1_o = np.dot((np.dot(x_prev, params[_p(prefix, 'Wa_o')]) * params[_p(prefix, 'yWb_o')]), params[_p(prefix, 'Wc_o')].T)
            tmp1_c = np.dot((np.dot(x_prev, params[_p(prefix, 'Wa_c')]) * params[_p(prefix, 'yWb_c')]), params[_p(prefix, 'Wc_c')].T)
            if x_prev_id >= 0: #not for -1 which is sent start
                params[_p(prefix, 'cacheX')][x_prev_id] = ((tmp1_i, tmp1_f, tmp1_o, tmp1_c))

        if h_prev_id >= 0 and h_prev_id in params[_p(prefix, 'cacheH')]:
            tmp2_i = params[_p(prefix, 'cacheH')][h_prev_id][0]
            tmp2_f = params[_p(prefix, 'cacheH')][h_prev_id][1]
            tmp2_o = params[_p(prefix, 'cacheH')][h_prev_id][2]
            tmp2_c = params[_p(prefix, 'cacheH')][h_prev_id][3]
        else:
            tmp2_i = np.dot((np.dot(h_prev, params[_p(prefix, 'Ua_i')]) * params[_p(prefix, 'yUb_i')]), params[_p(prefix, 'Uc_i')].T)
            tmp2_f = np.dot((np.dot(h_prev, params[_p(prefix, 'Ua_f')]) * params[_p(prefix, 'yUb_f')]), params[_p(prefix, 'Uc_f')].T)
            tmp2_o = np.dot((np.dot(h_prev, params[_p(prefix, 'Ua_o')]) * params[_p(prefix, 'yUb_o')]), params[_p(prefix, 'Uc_o')].T)
            tmp2_c = np.dot((np.dot(h_prev, params[_p(prefix, 'Ua_c')]) * params[_p(prefix, 'yUb_c')]), params[_p(prefix, 'Uc_c')].T)
            if h_prev_id >= 0: #not for -1 which is sent start
                params[_p(prefix, 'cacheH')][h_prev_id] = ((tmp2_i, tmp2_f, tmp2_o, tmp2_c))

        preact_i = tmp1_i + tmp2_i + params[_p(prefix, 'b_i')]
        preact_f = tmp1_f + tmp2_f + params[_p(prefix, 'b_f')]
        preact_o = tmp1_o + tmp2_o + params[_p(prefix, 'b_o')]
        preact_c = tmp1_c + tmp2_c + params[_p(prefix, 'b_c')]

        i = sigmoid(preact_i)
        f = sigmoid(preact_f)
        o = sigmoid(preact_o)
        c = np.tanh(preact_c)

        c = f * c_prev + i * c
        h = o * np.tanh(c)

        Vhid = params['vWemb']
        y0 = np.dot(h, Vhid) + params['bhid']

        return y0, h, c
    
    # calculate the prob. of next word using ensemble
    p0 = 0.
    num = len(params_set)
    h0_set = []
    c0_set = []
    for params in params_set:
        z0 =np.dot(z,params['C0'])
        h0 = np.zeros((n_h,))
        c0 = np.zeros((n_h,))
        (y0, h0, c0) = _step_set(z0, h0, c0, -1, -1, params) 
        h0_set.append(h0)
        c0_set.append(c0)
        maxy0 = np.amax(y0)
        e0 = np.exp(y0 - maxy0) # for numerical stability shift into good numerical range
        p0 = p0 + e0 / np.sum(e0)
    
    p0 = p0 / num
    y0 = np.log(1e-20 + p0) # and back to log domain

    beams = []
    nsteps = 1
    # generate the first word
    top_indices = np.argsort(-y0)  # we do -y because we want decreasing order

    # perform BEAM search.
    for i in range(beam_size):
        wordix = top_indices[i]
        # log probability, indices of words predicted in this beam so far, and the hidden and cell states
        beams.append((y0[wordix], [wordix], h0_set, c0_set, 0)) #add state id, too, e.g., 0 here
    # generate the rest n words
    while True:
        beam_candidates = []
        for params in params_set:
            params[_p(prefix, 'cacheH')].clear() #need to clear H cache at every step, but don't need for X
        preStateId = 0
        for b in beams:
            ixprev = b[1][-1] if b[1] else 0 # start off with the word where this beam left off
            if ixprev == 0 and b[1]:
                # this beam predicted end token. Keep in the candidates but don't expand it out any more
                beam_candidates.append(b)
                continue
            ihprev = b[4]
            
            h1_set = []
            c1_set = []
            p1 = 0.
            for ii in range(num):
                (y1, h1, c1) = _step_set(params['Wemb'][ixprev], b[2][ii], b[3][ii], ixprev, ihprev, params_set[ii])
                h1_set.append(h1)
                c1_set.append(c1)
                y1 = y1.ravel() # make into 1D vector
                maxy1 = np.amax(y1)
                e1 = np.exp(y1 - maxy1) # for numerical stability shift into good numerical range
                p1 = p1 + e1 / np.sum(e1)
            
            p1 = p1/num
            y1 = np.log(1e-20 + p1) # and back to log domain
            top_indices = np.argsort(-y1)  # we do -y because we want decreasing order

            for i in range(beam_size):
                wordix = top_indices[i]
                beam_candidates.append((b[0] + y1[wordix], b[1] + [wordix], h1_set, c1_set, preStateId))

            preStateId = preStateId + 1

        beam_candidates.sort(reverse = True) # decreasing order
        beams = beam_candidates[:beam_size] # truncate to get new beams
        nsteps += 1
        if nsteps >= max_step: # bad things are probably happening, break out
            break

    # strip the intermediates, only keep ppl and wordids
    predictions = [(b[0], b[1]) for b in beams]

    return predictions

def generate(z_emb, y_emb, params_set, beam_size, max_step):

    predset = []
    #print("count how many captions we have generated...")
    prefix='encoder_lstm'
    #params_set extension
    params_set_ext = []
    for params in params_set:
        params['vWemb'] = np.dot(params['Vhid'],params['Wemb'].T)
        params[_p(prefix, 'cacheX')] = OrderedDict()
        params[_p(prefix, 'cacheH')] = OrderedDict()
        params_set_ext.append(params)
    del params_set
     
    #print('start captioning generation @ ')
    #print(datetime.datetime.now().time())
    for i in range(len(z_emb)):
        y = y_emb[i]
        params_set = []
        for params in params_set_ext:        
            params[_p(prefix, 'yWb_i')] = np.dot(y, params[_p(prefix, 'Wb_i')])
            params[_p(prefix, 'yWb_f')] = np.dot(y, params[_p(prefix, 'Wb_f')])
            params[_p(prefix, 'yWb_o')] = np.dot(y, params[_p(prefix, 'Wb_o')])
            params[_p(prefix, 'yWb_c')] = np.dot(y, params[_p(prefix, 'Wb_c')])
            params[_p(prefix, 'yUb_i')] = np.dot(y, params[_p(prefix, 'Ub_i')])
            params[_p(prefix, 'yUb_f')] = np.dot(y, params[_p(prefix, 'Ub_f')])
            params[_p(prefix, 'yUb_o')] = np.dot(y, params[_p(prefix, 'Ub_o')])
            params[_p(prefix, 'yUb_c')] = np.dot(y, params[_p(prefix, 'Ub_c')])
            params[_p(prefix, 'cacheX')].clear()
            params[_p(prefix, 'cacheH')].clear()
            params_set.append(params)

        pred = predict(z_emb[i], params_set, beam_size, max_step)
        predset.append(pred)
        #print('.'),

    #print(' ')
    #print('end @ ')
    #print(datetime.datetime.now().time())
    
    return predset

    
def caption_generation(N, img_feats, tag_feats, beam_size =5, max_step = 20):
    path = './pretrained_model/params_'
    param_list = range(N) # define how many ensembles to use
    params_set = load_params(path, param_list)
    
    predset = generate(img_feats, tag_feats, params_set, beam_size=beam_size, max_step=max_step)
    
    x = pickle.load(open("./pretrained_model/caption_vocab.p","rb"))
    caption_ixtoword = x[1]
    del x

    predtext = []
    for sent in predset:
        rev = []
        sen = sent[0]
        smal = []
        for w in sen[1]:
            smal.append(caption_ixtoword[w])
        smal.pop() #remove the last '.'
        rev.append(' '.join(smal))
        predtext.append(rev)
    return predtext

    
