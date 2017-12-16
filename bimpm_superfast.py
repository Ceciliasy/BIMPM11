
# coding: utf-8

# In[3]:

import numpy as np
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]

        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

glove=loadGloveModel('data/wordvec.txt')


# In[4]:

import pandas as pd
import string
quora=pd.read_csv('data/train.tsv',sep='\t',header=None,index_col=0)
quora.columns=['question1','question2','ID']
quora['is_duplicate']=quora.index.astype('int')
quora.index=quora['ID']
quora=quora.drop('ID',axis=1)


# In[5]:

quora_dev=pd.read_csv('data/dev.tsv',sep='\t',header=None,index_col=0)
quora_dev.columns=['question1','question2','ID']
quora_dev['is_duplicate']=quora_dev.index.astype('int')
quora_dev.index=quora_dev['ID']
quora_dev=quora_dev.drop('ID',axis=1)


quora_test=pd.read_csv('data/test.tsv',sep='\t',header=None,index_col=0)
quora_test.columns=['question1','question2','ID']
quora_test['is_duplicate']=quora_test.index.astype('int')
quora_test.index=quora_test['ID']
quora_test=quora_test.drop('ID',axis=1)


# In[6]:

import pandas as pd
import string
#quora=pd.read_csv('quora_duplicate_questions.tsv',sep='\t',header=0,index_col=0)
def clean(quora):
    quora=quora.dropna(axis=0)

    #translator={s: None for s in string.punctuation+string.digits}

    drop_index=[]
    k=0
    for index, row in quora.iterrows():
        p=row['question1']
        q=row['question2']

    #     p=p.translate(p.maketrans(translator)).lower().strip()
    #     q=q.translate(q.maketrans(translator)).lower().strip()

        p=p.lower().strip()
        q=q.lower().strip()
        if (p=='')|(q==''):
            print(index)
            drop_index.append(index)
        if (len(p.split())<2)|((len(q.split())<2)):
            k+=1
            print(k)
            drop_index.append(index)

    quora=quora.drop(drop_index)
    return quora


# In[7]:

quora=clean(quora)
quora_dev=clean(quora_dev)
quora_test=clean(quora_test)


# In[8]:




class qpair():

    def __init__(self, p,q,p_mat,q_mat,p_character,q_character, label,p_word_lengths,q_word_lengths,p_word_count,q_word_count):
        self.p = p
        self.q = q
        self.p_mat = p_mat
        self.q_mat=q_mat
        self.p_character=p_character
        self.q_character=q_character
        self.label=label
        self.p_word_lengths=p_word_lengths
        self.q_word_lengths=q_word_lengths
        self.p_word_count=p_word_count
        self.q_word_count=q_word_count

    def set_q_mat(self, q_mat):
        self.q_mat=q_mat
    def set_p_mat(self,p_1mat):
        self.p_mat=p_mat
def get_glove(sentence):
    vector_list = np.array(list(map(lambda t: np.array(glove[t]) if t in glove else np.random.rand(300),sentence.split())))
    return vector_list
def encode_character(word):
    c_index=[]
    for c in word:
        c_index.append(ord(c)-96)
    c_index=np.array(c_index)
    c_index[c_index>25]=27
    c_index[c_index<0]=27
    return (c_index)

def load_set(quora):
    qpair_set=[]


    #translator={s: None for s in string.punctuation+string.digits}

    for index, row in quora.iterrows():
        p=row['question1']
        q=row['question2']
        #print(q)
    #     p=p.translate(p.maketrans(translator)).lower()
    #     q=q.translate(q.maketrans(translator)).lower()
        p=p.lower().strip()
        q=q.lower().strip()
        label=row['is_duplicate']
        q_mat=get_glove(q)
        p_mat=get_glove(p)
        #print(q.split())
        #print(p)
        p_character=np.concatenate(np.array(list(map(lambda w: encode_character(w),p.split())))).astype('int')
        q_character=np.concatenate(np.array(list(map(lambda w: encode_character(w),q.split())))).astype('int')
        p_w_lengths=(np.cumsum(np.array((list(map(lambda x:len(x),p.split())))))-1).astype('int')
        q_w_lengths=(np.cumsum(np.array((list(map(lambda x:len(x),q.split())))))-1).astype('int')
        new_pair=qpair(p,q,p_mat,q_mat, p_character,q_character,label,p_w_lengths,q_w_lengths,len(p_mat),len(q_mat))
        qpair_set.append(new_pair)

    print("set loaded!")
    return qpair_set


# In[9]:

qpair_set=load_set(quora)
v_qpair_set=load_set(quora_dev)
t_qpair_set=load_set(quora_test)




def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """

    data_size = len(data)
    num_batches_per_epoch = int((len(data))/batch_size)
    #num_batches_per_epoch=10
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
#         if shuffle:
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]




def eval_iter(data, batch_size,limit):
    """
    Generates a batch iterator for a dataset.
    """

    data_size = len(data)
    num_batches = int((len(data))/batch_size)


    for batch_num in range(min(num_batches,limit)):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield data[start_index:end_index]


# In[15]:

def bceloss(o,t):
    return -sum((t[:,0]*np.log(o[:,0]))+(1-t[:,0])*(np.log(1-o[:,0]))+(t[:,1]*np.log(o[:,1]))+((1-t[:,1])*np.log(1-o[:,1]))+1e-7)/t.shape[1]/t.shape[0]


def evaluate(model, e_iter):
    model.eval()
    correct = 0
    total = 0
    loss=0
    k=0
    criterion = nn.BCELoss()
    for input_pairs in e_iter:
        print('eval:')
        print(k)

        label=np.array(list(map(lambda o:o.label, input_pairs)))

        label=(torch.from_numpy(label).long().cuda())
        outputs = model(input_pairs)

        _, predicted = torch.max(outputs.data, 1)

        total += label.shape[0]
        correct += (predicted == label).sum()
        # total += 1
        # correct += 1


        outputs=outputs.cpu().data.numpy()
        label=list(map(lambda o:o.label, input_pairs))
        label=np.vstack([np.array(label),1-np.array(label)]).T
        loss+=bceloss(outputs,label)

        k+=1
    return (correct / float(total),loss/ k)


# In[16]:

def training_loop(batch_size, num_epochs, model, optim, data_iter,total_len):
    step = 0
    epoch = 0
    losses = []
    accs=[]
    dev_losses=[]
    test_losses=[]
    test_accs=[]
    total_batches = int(total_len / batch_size)
    criterion = nn.BCELoss()
    optimizer = optim
    while epoch <= num_epochs:
        model.train()
        input_qpairs = next(data_iter)
        label=list(map(lambda o:o.label, input_qpairs))
        label=np.vstack([np.array(label),1-np.array(label)]).T
        label=Variable(torch.from_numpy(label).float().cuda())
        #label=Variable(torch.from_numpy(label).float())
        model.zero_grad()


        outputs=model(input_qpairs)
        #print(outputs)
        #print(label)
        loss=criterion(outputs, label)
        losses.append(loss.data[0])
        loss.backward()
        optimizer.step()
        print('Step:',step)
        print('Epoch',epoch)
        print('loss',loss.data[0])
        if step % total_batches == 0:
            epoch += 1

            if step> 0:
                e_iter=eval_iter(v_qpair_set,batch_size,500000)
                score,dev_loss=evaluate(model, e_iter)
                print( "Epoch:", (epoch), 'ACC:',score )
                print("Dev loss:",dev_loss)
                accs.append(score)
                dev_losses.append(dev_loss)

            if (epoch % 3==0):
                e_iter=eval_iter(t_qpair_set,batch_size,500000)
                score,dev_loss=evaluate(model, e_iter)
                test_accs.append(score)
                test_losses.append(dev_loss)
                print('TEst',score,dev_loss)
        step += 1
    np.savetxt('dev_accs.txt',accs)
    np.savetxt('dev_losses.txt',dev_losses)
    np.savetxt('test_accs.txt',test_accs)
    np.savetxt('test_losses.txt',test_dev_losses)


# In[17]:

def get_padded_2d(longlist):
    max_length=max(list(map(len,longlist)))
    nrow=len(longlist)
    array=np.zeros((nrow,max_length))
    for i in range(nrow):
        array[i,:len(longlist[i])]=longlist[i]
    return array




# In[18]:

def get_padded_3d(longlist):
    max_length=max(list(map(len,longlist)))

    d1=len(longlist)
    d2=max_length
    d3=len(longlist[0][0])
    array=np.zeros((d1,d2,d3))
    for i in range(d1):
        array[i,:len(longlist[i]),:]=longlist[i]
    return array


# In[1]:

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
#from .glove import qpair




class Bimpm(nn.Module):

    def __init__(self, embedding_dim,c_embedding_dim, c_l_hidden, lstm_hidden,dropout_ratio,l_hidden_size,L,batch_size):
        super(Bimpm, self).__init__()

        self.lstm_hidden=lstm_hidden
        self.c_l_hidden=c_l_hidden
        self.batch_size=batch_size
        self.c_embedding=nn.Embedding(28,c_embedding_dim)
        self.p_bilstm=nn.LSTM(embedding_dim,lstm_hidden,1,dropout=dropout_ratio,bidirectional=True)
        self.q_bilstm=nn.LSTM(embedding_dim,lstm_hidden,1,dropout=dropout_ratio,bidirectional=True)

        self.p_c_lstm=nn.LSTM(c_embedding_dim,c_l_hidden,1,dropout=dropout_ratio)
        self.q_c_lstm=nn.LSTM(c_embedding_dim,c_l_hidden,1,dropout=dropout_ratio)


        self.match_w_1_f=nn.Embedding(L,lstm_hidden)
        self.match_w_2_f=nn.Embedding(L,lstm_hidden)
        self.match_w_3_f=nn.Embedding(L,lstm_hidden)
        self.match_w_4_f=nn.Embedding(L,lstm_hidden)

        self.match_w_1_b=nn.Embedding(L,lstm_hidden)
        self.match_w_2_b=nn.Embedding(L,lstm_hidden)
        self.match_w_3_b=nn.Embedding(L,lstm_hidden)
        self.match_w_4_b=nn.Embedding(L,lstm_hidden)

        self.match_w_1_f_q=nn.Embedding(L,lstm_hidden)
        self.match_w_2_f_q=nn.Embedding(L,lstm_hidden)
        self.match_w_3_f_q=nn.Embedding(L,lstm_hidden)
        self.match_w_4_f_q=nn.Embedding(L,lstm_hidden)

        self.match_w_1_b_q=nn.Embedding(L,lstm_hidden)
        self.match_w_2_b_q=nn.Embedding(L,lstm_hidden)
        self.match_w_3_b_q=nn.Embedding(L,lstm_hidden)
        self.match_w_4_b_q=nn.Embedding(L,lstm_hidden)

        self.pmat_bilstm=nn.LSTM(8*L,lstm_hidden,1,dropout=dropout_ratio,bidirectional=True)
        self.qmat_bilstm=nn.LSTM(8*L,lstm_hidden,1,dropout=dropout_ratio,bidirectional=True)

        self.L1 = nn.Linear(4*lstm_hidden,l_hidden_size)
        self.L2 = nn.Linear(l_hidden_size,2)
    def forward(self,input_batch):
        k=0

        p_w=get_padded_3d([pair.p_mat for pair in input_batch])
        q_w=get_padded_3d([pair.q_mat for pair in input_batch])

        p_char=get_padded_2d([pair.p_character for pair in input_batch])
        q_char=get_padded_2d([pair.q_character for pair in input_batch])

        p_word_lengths=get_padded_2d([pair.p_word_lengths for pair in input_batch])
        q_word_lengths=get_padded_2d([pair.q_word_lengths for pair in input_batch])

        p_word_count=np.array([pair.p_word_count for pair in input_batch])
        q_word_count=np.array([pair.q_word_count for pair in input_batch])


#         p_w=Variable(torch.from_numpy(p_w).float())
#         q_w=Variable(torch.from_numpy(q_w).float())
        p_w=Variable(torch.from_numpy(p_w).float().cuda())
        q_w=Variable(torch.from_numpy(q_w).float().cuda())
        p_w=p_w.permute(1,0,2)
        q_w=q_w.permute(1,0,2)
#         print('p_w')
#         print(p_w.size())
#         p_char=Variable(torch.from_numpy(p_char).long())
#         q_char=Variable(torch.from_numpy(q_char).long())
#         p_word_lengths=Variable(torch.from_numpy(p_word_lengths).long())
#         q_word_lengths=Variable(torch.from_numpy(q_word_lengths).long())

        p_char=Variable(torch.from_numpy(p_char).long().cuda())
        q_char=Variable(torch.from_numpy(q_char).long().cuda())
        p_word_lengths=Variable(torch.from_numpy(p_word_lengths).long().cuda())
        q_word_lengths=Variable(torch.from_numpy(q_word_lengths).long().cuda())
        p_char_embed=self.c_embedding(p_char)
        q_char_embed=self.c_embedding(q_char)
        p_char_embed=p_char_embed.permute(1,0,2)
        q_char_embed=q_char_embed.permute(1,0,2)
#         print('p_char_embed')
#         print(p_char_embed.size())

#         c_h0 = Variable(torch.randn(1, self.batch_size, self.c_l_hidden))
#         c_c0 = Variable(torch.randn(1, self.batch_size, self.c_l_hidden))

        c_h0 = Variable(torch.randn(1, self.batch_size, self.c_l_hidden).cuda())
        c_c0 = Variable(torch.randn(1, self.batch_size, self.c_l_hidden).cuda())
        p_c_output,(p_c_h_n,p_c_c_n)=self.p_c_lstm(p_char_embed,(c_h0,c_c0))
        q_c_output,(q_c_h_n,q_c_c_n)=self.q_c_lstm(q_char_embed,(c_h0,c_c0))

#         print('p_c_output')
#         print(p_c_output.size())
        p_word_lengths=p_word_lengths.permute(1,0).unsqueeze(2)
        p_word_lengths=p_word_lengths.expand(p_word_lengths.size()[0],p_word_lengths.size()[1],p_c_output.size()[2])
        q_word_lengths=q_word_lengths.permute(1,0).unsqueeze(2)
        q_word_lengths=q_word_lengths.expand(q_word_lengths.size()[0],q_word_lengths.size()[1],q_c_output.size()[2])

#         print('p_word_lengths')
#         print(p_word_lengths.size())

        p_c_embed=torch.gather(p_c_output,0,p_word_lengths)
        q_c_embed=torch.gather(q_c_output,0,q_word_lengths)
#         print('p_c_embed')
#         print(p_c_embed.size())
        #p_c_embed=torch.squeeze(p_c_embed,dim=1)
        #q_c_embed=torch.squeeze(q_c_embed,dim=1)

#             print('p_w')
#             print(p_w.size())
#             print('p_c_embed')
#             print(p_c_embed.size())
#             print('p_c_output')
#             print(p_c_output.size())
        p_input=torch.cat((p_w,p_c_embed), 2)
        q_input=torch.cat((q_w,q_c_embed), 2)
#         print('p_input')
#         print(p_input.size())


#         h0_b = Variable(torch.randn(2, self.batch_size, self.lstm_hidden))
#         c0_b = Variable(torch.randn(2, self.batch_size, self.lstm_hidden))

        h0_b = Variable(torch.randn(2, self.batch_size, self.lstm_hidden).cuda())
        c0_b = Variable(torch.randn(2, self.batch_size, self.lstm_hidden).cuda())



        p_output,(p_h_n,p_c_n)=self.p_bilstm(p_input,(h0_b,c0_b))
        q_output,(q_h_n,q_c_n)=self.q_bilstm(q_input,(h0_b,c0_b))


#         print('p_output')
#         print(p_output.size())


        big_p_output_forward=p_output[:,:,:self.lstm_hidden]
        big_p_output_backward=p_output[:,:,self.lstm_hidden:]
        big_q_output_forward=q_output[:,:,:self.lstm_hidden]
        big_q_output_backward=q_output[:,:,self.lstm_hidden:]



        p_output_forward=big_p_output_forward.permute(2,1,0)
        p_output_backward=big_p_output_backward.permute(2,1,0)
        q_output_forward=big_q_output_forward.permute(2,1,0)
        q_output_backward=big_q_output_backward.permute(2,1,0)


        m_p=matching(p_output_forward,p_output_backward,q_output_forward,q_output_backward,self.match_w_1_f
                    ,self.match_w_2_f,self.match_w_3_f,self.match_w_4_f,self.match_w_1_b,
                     self.match_w_2_b,self.match_w_3_b,self.match_w_4_b)
        m_q=matching(q_output_forward,q_output_backward,p_output_forward,p_output_backward,
                     self.match_w_1_f_q,self.match_w_2_f_q,self.match_w_3_f_q,self.match_w_4_f_q,
                     self.match_w_1_b_q,self.match_w_2_b_q,self.match_w_3_b_q,self.match_w_4_b_q)


#         m_h0 = Variable(torch.randn(2, self.batch_size, self.lstm_hidden))
#         m_c0 = Variable(torch.randn(2, self.batch_size, self.lstm_hidden))

        m_h0 = Variable(torch.randn(2, self.batch_size, self.lstm_hidden).cuda())
        m_c0 = Variable(torch.randn(2, self.batch_size, self.lstm_hidden).cuda())
        #m_p=m_p.unsqueeze(1)
        #m_q=m_q.unsqueeze(1)
        pmat_output,(pmat_h_n,pmat_c_n)=self.pmat_bilstm(m_p,(m_h0,m_c0))
        qmat_output,(qmat_h_n,qmat_c_n)=self.qmat_bilstm(m_q,(m_h0,m_c0))


        pmat_0=pmat_h_n[1]
        pmat_n=pmat_h_n[0]

        qmat_0=qmat_h_n[1]
        qmat_n=qmat_h_n[0]

        aggresult=torch.cat((pmat_0,pmat_n,qmat_0,qmat_n),dim=1).squeeze(0)

        output1 = self.L1(aggresult)
        output2 = self.L2(output1)
        output=nn.functional.sigmoid(output2)

#         if k==0:
#             outputs=output
#         else:

#             outputs=torch.cat((outputs,output),1)
#         k+=1
#         outputs=outputs.permute(1,0)
        outputs=output
        return outputs


# In[2]:

from torch.nn import CosineSimilarity
def matching(p_output_forward,p_output_backward,q_output_forward,q_output_backward,match_w_1_f,match_w_2_f
             ,match_w_3_f,match_w_4_f,match_w_1_b,match_w_2_b,match_w_3_b,match_w_4_b):
    for l in range(L):
        current_l=Variable(torch.from_numpy(np.array([l])).cuda())
#         current_l=Variable(torch.from_numpy(np.array([l])))

        #full_matching forward
        current_w=match_w_1_f(current_l).permute(1,0)
        current_w=current_w.expand(current_w.size()[0],p_output_forward.size()[1])
        current_w=current_w.unsqueeze(2)
        current_w=current_w.expand(current_w.size()[0],current_w.size()[1],p_output_forward.size()[2])

        q_last_h=q_output_forward[:,:,q_output_forward.size()[2]-1]
        q_last_h=q_last_h.unsqueeze(2)

        q_last_h=q_last_h.expand(q_last_h.size()[0],q_last_h.size()[1],p_output_forward.size()[2])

        V_1=torch.mul(current_w,p_output_forward)
        V_2=torch.mul(current_w,q_last_h)

        cos = CosineSimilarity(dim=0, eps=1e-6)
        sim=cos(V_1,V_2)

        if l==0:
            m_1_forward=sim.permute(1,0).unsqueeze(2)
        else:

            m_1_forward=torch.cat((m_1_forward,sim.permute(1,0).unsqueeze(2)),2)



        #full_matching backward
        current_w=match_w_1_b(current_l).permute(1,0)
        current_w=current_w.expand(current_w.size()[0],p_output_backward.size()[1])
        current_w=current_w.unsqueeze(2)
        current_w=current_w.expand(current_w.size()[0],current_w.size()[1],p_output_backward.size()[2])
        q_last_h=q_output_backward[:,:,q_output_backward.size()[2]-1]
        q_last_h=q_last_h.unsqueeze(2)

        q_last_h=q_last_h.expand(q_last_h.size()[0],q_last_h.size()[1],p_output_backward.size()[2])

        V_1=torch.mul(current_w,p_output_backward)
        V_2=torch.mul(current_w,q_last_h)

        cos = CosineSimilarity(dim=0, eps=1e-6)
        sim=cos(V_1,V_2)


        if l==0:
            m_1_backward=sim.permute(1,0).unsqueeze(2)
        else:
            m_1_backward=torch.cat((m_1_backward,sim.permute(1,0).unsqueeze(2)),2)


        #################
         #max_matching forward


        current_w=match_w_2_f(current_l).permute(1,0)
        current_w=current_w.expand(current_w.size()[0],p_output_forward.size()[1])
        current_w=current_w.unsqueeze(2)
        current_w_p=current_w.expand(current_w.size()[0],current_w.size()[1],p_output_forward.size()[2])
        current_w_q=current_w.expand(current_w.size()[0],current_w.size()[1],q_output_forward.size()[2])


        V_1=torch.mul(current_w,p_output_forward)
        V_2=torch.mul(current_w,q_output_forward)

        V_1=V_1.unsqueeze(-1)
        V_1=V_1.expand(V_1.size()[0],V_1.size()[1],V_1.size()[2],q_output_forward.size()[2])

        V_2=V_2.unsqueeze(-1)
        V_2=V_2.expand(V_2.size()[0],V_2.size()[1],V_2.size()[2],p_output_forward.size()[2])
        V_2=V_2.permute(0,1,3,2)

        cos = CosineSimilarity(dim=0, eps=1e-6)
        sim=cos(V_1,V_2)


        sim=torch.max(sim,2)[0]


        if l==0:
            m_2_forward=sim.permute(1,0).unsqueeze(2)
        else:
            m_2_forward=torch.cat((m_2_forward,sim.permute(1,0).unsqueeze(2)),2)


         #max_matching backward
        current_w=match_w_2_b(current_l).permute(1,0)
        current_w=current_w.expand(current_w.size()[0],p_output_backward.size()[1])
        current_w=current_w.unsqueeze(2)
        current_w_p=current_w.expand(current_w.size()[0],current_w.size()[1],p_output_backward.size()[2])
        current_w_q=current_w.expand(current_w.size()[0],current_w.size()[1],q_output_backward.size()[2])

#                     print('q_output_forward')
#                     print(q_output_forward.size())
#                     print('p_output_forward')
#                     print(p_output_forward.size())
        V_1=torch.mul(current_w,p_output_backward)
        V_2=torch.mul(current_w,q_output_backward)

        V_1=V_1.unsqueeze(-1)
        V_1=V_1.expand(V_1.size()[0],V_1.size()[1],V_1.size()[2],q_output_backward.size()[2])

        V_2=V_2.unsqueeze(-1)
        V_2=V_2.expand(V_2.size()[0],V_2.size()[1],V_2.size()[2],p_output_backward.size()[2])
        V_2=V_2.permute(0,1,3,2)

        cos = CosineSimilarity(dim=0, eps=1e-6)
        sim=cos(V_1,V_2)
#                     print(sim.size())

        sim=torch.max(sim,2)[0]

        if l==0:
            m_2_backward=sim.permute(1,0).unsqueeze(2)
        else:
            m_2_backward=torch.cat((m_2_backward,sim.permute(1,0).unsqueeze(2)),2)



##################
        #Attentive -Matching forward


        current_w = match_w_3_f(current_l).permute(1,0)
        V_1 = p_output_forward
        V_2 = q_output_forward

        V_1=V_1.unsqueeze(-1)
        V_1=V_1.expand(V_1.size()[0],V_1.size()[1],V_1.size()[2],q_output_forward.size()[2])

        V_2=V_2.unsqueeze(-1)
        V_2=V_2.expand(V_2.size()[0],V_2.size()[1],V_2.size()[2],p_output_forward.size()[2])
        V_2=V_2.permute(0,1,3,2)
        cos=CosineSimilarity(dim=0, eps=1e-6)

        att=cos(V_1,V_2)

        att_sum=torch.sum(att,2)
        att_sum=att_sum.unsqueeze(2)
        att_sum=att_sum.expand(att_sum.size()[0],att_sum.size()[1],att.size()[2])

        att=att/att_sum
        att=att.unsqueeze(3)
        att=att.expand(att.size()[0],att.size()[1],att.size()[2],V_2.size()[0]).permute(3,0,1,2)

        hmean= torch.sum(att*V_2,3)

        current_w=current_w.expand(current_w.size()[0],p_output_forward.size()[1])
        current_w=current_w.unsqueeze(2)
        current_w=current_w.expand(current_w.size()[0],current_w.size()[1],p_output_forward.size()[2])

        V_1=torch.mul(current_w,p_output_forward)
        V_2=torch.mul(current_w,hmean)


        cos=CosineSimilarity(dim=0, eps=1e-6)
        sim=cos(V_1,V_2)

        if l==0:
            m_3_forward=sim.permute(1,0).unsqueeze(2)
        else:
            m_3_forward=torch.cat((m_3_forward,sim.permute(1,0).unsqueeze(2)),2)





        #Attentive -Matching backward


        current_w = match_w_3_f(current_l).permute(1,0)
        V_1 = p_output_backward
        V_2 = q_output_backward

        V_1=V_1.unsqueeze(-1)
        V_1=V_1.expand(V_1.size()[0],V_1.size()[1],V_1.size()[2],q_output_backward.size()[2])

        V_2=V_2.unsqueeze(-1)
        V_2=V_2.expand(V_2.size()[0],V_2.size()[1],V_2.size()[2],p_output_backward.size()[2])
        V_2=V_2.permute(0,1,3,2)

        cos=CosineSimilarity(dim=0, eps=1e-6)

        att=cos(V_1,V_2)

        att_sum=torch.sum(att,2)
        att_sum=att_sum.unsqueeze(2)
        att_sum=att_sum.expand(att_sum.size()[0],att_sum.size()[1],att.size()[2])
        att=att/att_sum
        att=att.unsqueeze(3)
        att=att.expand(att.size()[0],att.size()[1],att.size()[2],V_2.size()[0]).permute(3,0,1,2)

        hmean = torch.sum(att*V_2,3)

        current_w=current_w.expand(current_w.size()[0],p_output_backward.size()[1])
        current_w=current_w.unsqueeze(2)
        current_w=current_w.expand(current_w.size()[0],current_w.size()[1],p_output_backward.size()[2])

        V_1=torch.mul(current_w,p_output_backward)
        V_2=torch.mul(current_w,hmean)


        cos=CosineSimilarity(dim=0, eps=1e-6)
        sim=cos(V_1,V_2)

        if l==0:
            m_3_backward=sim.permute(1,0).unsqueeze(2)
        else:
            m_3_backward=torch.cat((m_3_backward,sim.permute(1,0).unsqueeze(2)),2)


        #Max-Attentive-Matching forward
        current_w = match_w_4_f(current_l).permute(1,0)

        V_1 = p_output_forward
        V_2 = q_output_forward

        V_1=V_1.unsqueeze(-1)
        V_1=V_1.expand(V_1.size()[0],V_1.size()[1],V_1.size()[2],q_output_forward.size()[2])

        V_2=V_2.unsqueeze(-1)
        V_2=V_2.expand(V_2.size()[0],V_2.size()[1],V_2.size()[2],p_output_forward.size()[2])
        V_2=V_2.permute(0,1,3,2)
        cos=CosineSimilarity(dim=0, eps=1e-6)

        att=cos(V_1,V_2)

        max_indices = torch.max(att,2)[1]
        max_indices=max_indices.unsqueeze(2)
        max_indices=max_indices.expand(max_indices.size()[0],max_indices.size()[1],q_output_forward.size()[0])
        max_indices=max_indices.permute(2,0,1)

        hmax = torch.gather(q_output_forward,2,max_indices)

        current_w=current_w.expand(current_w.size()[0],p_output_forward.size()[1])
        current_w=current_w.unsqueeze(2)
        current_w=current_w.expand(current_w.size()[0],current_w.size()[1],p_output_forward.size()[2])

        V_1=torch.mul(current_w,p_output_forward)
        V_2=torch.mul(current_w,hmax)

        cos=CosineSimilarity(dim=0, eps=1e-6)
        sim=cos(V_1,V_2)


        if l==0:
            m_4_forward=sim.permute(1,0).unsqueeze(2)
        else:
            m_4_forward=torch.cat((m_4_forward,sim.permute(1,0).unsqueeze(2)),2)

        #Attentive -Matching backward
        current_w = match_w_4_f(current_l).permute(1,0)

        V_1 = p_output_backward
        V_2 = q_output_backward

        V_1=V_1.unsqueeze(-1)
        V_1=V_1.expand(V_1.size()[0],V_1.size()[1],V_1.size()[2],q_output_backward.size()[2])

        V_2=V_2.unsqueeze(-1)
        V_2=V_2.expand(V_2.size()[0],V_2.size()[1],V_2.size()[2],p_output_backward.size()[2])
        V_2=V_2.permute(0,1,3,2)

        cos=CosineSimilarity(dim=0, eps=1e-6)

        att=cos(V_1,V_2)


        max_indices = torch.max(att,2)[1]
        max_indices=max_indices.unsqueeze(2)
        max_indices=max_indices.expand(max_indices.size()[0],max_indices.size()[1],q_output_backward.size()[0])
        max_indices=max_indices.permute(2,0,1)

        hmax = torch.gather(q_output_backward,2,max_indices)

        current_w=current_w.expand(current_w.size()[0],p_output_backward.size()[1])
        current_w=current_w.unsqueeze(2)
        current_w=current_w.expand(current_w.size()[0],current_w.size()[1],p_output_backward.size()[2])


        V_1=torch.mul(current_w,p_output_backward)
        V_2=torch.mul(current_w,hmax)

        cos=CosineSimilarity(dim=0, eps=1e-6)
        sim=cos(V_1,V_2)


        if l==0:
            m_4_backward=sim.permute(1,0).unsqueeze(2)
        else:
            m_4_backward=torch.cat((m_4_backward,sim.permute(1,0).unsqueeze(2)),2)

    m=torch.cat((m_1_forward,m_1_backward,m_2_forward,m_2_backward,m_3_forward,m_3_backward,m_4_forward,m_4_backward),2)
    return m


# In[20]:

c_embedding_dim=20
embedding_dim=350
c_l_hidden=50
lstm_hidden=100
dropout_ratio=0.1
l_hidden_size=5
L=2
batch_size=16
num_epochs=20
# model=Bimpm(embedding_dim,c_embedding_dim, c_l_hidden, lstm_hidden, dropout_ratio,l_hidden_size,L,batch_size)
model=Bimpm(embedding_dim,c_embedding_dim, c_l_hidden, lstm_hidden, dropout_ratio,l_hidden_size,L,batch_size).cuda()


# In[21]:
qpair_set=qpair_set
data_iter=batch_iter(qpair_set,batch_size,num_epochs)
learning_rate=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_len=len(qpair_set)
training_loop(batch_size, num_epochs, model, optimizer, data_iter,total_len)
