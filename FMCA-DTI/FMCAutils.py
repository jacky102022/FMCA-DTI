import sys
# sys.path.append("../")
# from utils.utils_helper import *
from utils.file_helper import *
import deepsmiles
from decompose.feature import *
from decompose.decompose_brackets_helper import *
# doc2vec_config=doc2vec_params_config()
# word2vec_config=word2vec_config()
import glob

from utils.DataPrepare import get_kfold_data, shuffle_dataset

def DTI_dataset_config():
    config = {}
    config["BIOSNAP"] = "./data/BIOSNAP"
    config["DAVIS"] = "./data/DAVIS"
    config["BindingDB"] = "./data/BindingDB"
    config["human"] = "./data/Human"
    config["celegans"] = "./data/Celegans"
    return config

dataset_config=DTI_dataset_config()

def list2zip(data):
    zipped=[tuple(a) for a in data]
    datasetList=list(zip(*zipped))
    smiles=list(datasetList[0])
    protein=list(datasetList[1])
    label=list(datasetList[2])
    return smiles,protein,label

def assert_dir_exist(x):
    if not os.path.exists(x):
        os.makedirs(x)
def save_best_model(model, model_dir, best_epoch):
    # save parameters of trained model
    # print(model_dir)
    torch.save(model.state_dict(), model_dir + '{}.pkl'.format(best_epoch))
    files = glob.glob(model_dir + '*.pkl')
    # print()
    # delete models saved before
    for file in files:
        tmp = file.split('/')[-1]  # windows:\\  linux: /
        tmp = tmp.split('.')[0]
        # print(tmp)
        epoch_nb = int(tmp)
        if epoch_nb < best_epoch:
            os.remove(file)


def smiles2deepsmiles():
    for root_path in reversed(dataset_config["datasetDir"]):
        save_path=root_path+"/"+"original"
        input_path=root_path+"/"+"input"
        dataset=load_txt(save_path+"/"+"data_filter")
        f_dp=open(root_path+"/"+"original"+"/"+"data_filter_dp","w")
        print(save_path+"/"+"data_filter")
        converter=deepsmiles.Converter(rings=True,branches=True)
        # dataset_filter_deep=[]
        dp_s=[]
        dp_p=[]
        dp_l=[]
        for i,line in enumerate(dataset):
            s,p,l=line.strip().split(" ")
            try:
                s_en=converter.encode(s)
                s_de=converter.decode(s_en)
                dp_s.append(s_de)
                dp_p.append(p)
                dp_l.append(int(l))
                f_dp.write(" ".join([s_de,p,l]))
                # dataset_filter_deep.append([s_de,p,l])
            except deepsmiles.DecodeError as e:
                print("DecodeError! Error message was '%s'"%e.message)
        # write2txt(save_path+"/"+"data_filter_dp",dataset_filter_deep)
        np.save(input_path+"/"+"smiles_filter_dp",dp_s)
        np.save(input_path+"/"+"protein_filter_dp",dp_p)
        np.save(input_path+"/"+"label_filter_dp",dp_l)
        f_dp.close()


from  decompose.decompose_brackets_helper import *
import codecs
from subword_nmt.apply_bpe import BPE

def get_one_bcm(smiles,protein,label,decompose2="category",k=3):
    decompose_dataset_smiles=[]
    decompose_dataset_protein=[]
    for i,s in enumerate(smiles):
        # print(i)
        x=s
        try:
            mol=Chem.MolFromSmiles(s)
            s=Chem.MolToSmiles(mol)
        except:
            s=x
        tmp= extract_sub(s)
        decompose_dataset_smiles.append(tmp)
    if decompose2=="category":
        for p in protein:
            tmp=protein2category(p)
            tmp_gram=to_gram_no_overlap(tmp,k)
            decompose_dataset_protein.append(tmp_gram)
    elif decompose2=="":
        for p in protein:
            tmp_gram=to_gram_no_overlap(p,k)
            decompose_dataset_protein.append(tmp_gram)
    return decompose_dataset_smiles,decompose_dataset_protein,label

def get_one_recap(smiles,protein,label,decompose=mol2Recap,k=3):
    decompose_dataset_smiles = []
    decompose_dataset_protein = []
    function_frag=decompose
    for s in smiles:
        frag=function_frag(s)
        if frag:
            decompose_dataset_smiles.append(frag)
        else:
            decompose_dataset_smiles.append(s)
    for p in protein:
        p_category = protein2category(p)
        p_category_3_gram = to_gram_no_overlap(p_category,k)
        decompose_dataset_protein.append(p_category_3_gram)

    return decompose_dataset_smiles,decompose_dataset_protein,label

def  load_all_dataset(input_path,decompose2="category"):
    datasetSmiles=[]
    datasetProtein=[]
    datasetLabel=[]

    datasetAll=load_txt(input_path+"/"+"data")
    # print(len(datasetAll),len(datasetSmiles))
    for line in datasetAll:
        s,p,l=line.split( )#" "
        # print(i)
        x = s
        try:
            mol = Chem.MolFromSmiles(s)
            s = Chem.MolToSmiles(mol)
        except:
            s = x
        datasetSmiles.append(s)
        datasetProtein.append(p)
        datasetLabel.append(int(l))
    return datasetSmiles,datasetProtein,datasetLabel

def get_one_ch(smiles,protein,label,k=3):
    decompose_dataset_smiles=[]
    decompose_dataset_protein=[]
    for i, s in enumerate(smiles):
        decompose_dataset_smiles.append(list(s))
    for p in protein:
        decompose_dataset_protein.append(list(p))

    return decompose_dataset_smiles, decompose_dataset_protein, label

def load_frag_params(datasetSmiles, datasetProtein, datasetLabel):

    frag_set_d = list(set(l for s in datasetSmiles for l in s))
    frag_set_p = list(set(l for s in datasetProtein for l in s))

    frag_len_d = [len(d) for d in datasetSmiles]
    frag_len_p = [len(p) for p in datasetProtein]

    words2idx_d = dict(zip(frag_set_d, range(0, len(frag_set_d))))
    words2idx_p = dict(zip(frag_set_p, range(0, len(frag_set_p))))
    return frag_set_d, frag_set_p, frag_len_d, frag_len_p, words2idx_d, words2idx_p


def split_train_test_set(datasetSmiles, datasetProtein, datasetLabel,split_random=True):

    alldataset = [list(pair) for pair in zip(datasetSmiles, datasetProtein, datasetLabel)]

    # 将每个子列表转换为逗号分隔的字符串
    alldataset = [','.join(map(str, item)) for item in alldataset]

    '''shuffle data'''
    print("data shuffle")
    alldataset = shuffle_dataset(alldataset, 114514)
    '''split dataset to train&validation set and test set'''
    split_pos = len(alldataset) - int(len(alldataset) * 0.2)
    train_data_list = alldataset[0:split_pos]
    test_data_list = alldataset[split_pos:-1]
    print('Number of Train&Val set: {}'.format(len(train_data_list)))
    print('Number of Test set: {}'.format(len(test_data_list)))
    # train_dataset = [item.split(',') for item in train_data_list]
    # test_dataset = [item.split(',') for item in test_data_list]

    return train_data_list, test_data_list


def load_case_data():
    drug=load_txt("drug")
    target=load_txt("target")
    name_pair=[]
    data_pair=[]
    smiles=[]
    protein=[]
    for d in drug:
        d=d.strip().split(" ")
        names,s=d[0],d[1]
        for t in target:
            t=t.strip().split(" ")
            if "" in t:
                t.remove("")
            if t:
                namep,p=" ".join(t[:-1]),t[-1]
                name_pair.append([names,namep])
                data_pair.append([s,p])
                smiles.append(s)
                protein.append(p)
    return name_pair,data_pair,smiles,protein

def load_frag(train_dataset, valid_dataset, test_dataset,decompose,decompose_protein,unseen_smiles,k,split_random):
  
    trainSmiles, trainProtein, trainLabel = list2zip(train_dataset)
    valSmiles, valProtein, valLabel = list2zip(valid_dataset)
    testSmiles, testProtein, testLabel = list2zip(test_dataset)

    if decompose == "bcm":
        trainSmiles, trainProtein, trainLabel=get_one_bcm(trainSmiles,trainProtein,trainLabel,decompose2=decompose_protein,k=k)
        valSmiles,valProtein,valLabel=get_one_bcm(valSmiles,valProtein,valLabel,decompose2=decompose_protein,k=k)
        testSmiles,testProtein,testLabel=get_one_bcm(testSmiles,testProtein,testLabel,decompose2=decompose_protein,k=k)
   
    else:
        trainSmiles, trainProtein, trainLabel = get_one_ch(trainSmiles, trainProtein, trainLabel,k=k)
        valSmiles, valProtein, valLabel = get_one_ch(valSmiles, valProtein, valLabel,k=k)
        testSmiles, testProtein, testLabel = get_one_ch(testSmiles, testProtein, testLabel,k=k)

    frag_set_d, frag_set_p, frag_len_d, frag_len_p, words2idx_d, words2idx_p=load_frag_params(trainSmiles+valSmiles+testSmiles, trainProtein+valProtein+testProtein,trainLabel+valLabel+testLabel)

    root_path = dataset_config["DAVIS"]
    input_path = root_path + "/"


    return trainSmiles, trainProtein, trainLabel,valSmiles, valProtein, valLabel,testSmiles, testProtein, testLabel,frag_set_d, frag_set_p, frag_len_d, frag_len_p, words2idx_d, words2idx_p\

def split_unseen_drug(datasetSmiles, datasetProtein, datasetLabel):
    trainSmiles=[]
    trainProtein=[]
    trainLabel=[]
    valSmiles=[]
    valProtein=[]
    valLabel=[]
    testSmiles=[]
    testProtein=[]
    testLabel=[]

    smiles_set=list(set(datasetSmiles))
    train_size = int(0.8 * len(smiles_set))
    valid_size = int(0.1 * len(smiles_set))
    test_size = len(smiles_set) - train_size - valid_size

    trainSmilesSet=smiles_set[:train_size]
    valSmilesSet=smiles_set[train_size:valid_size+train_size]
    testSmilesSet=smiles_set[valid_size+train_size:-1]


    for i ,s in enumerate(datasetSmiles):
        if s in  trainSmilesSet:
            trainSmiles.append(s)
            trainProtein.append(datasetProtein[i])
            trainLabel.append(datasetLabel[i])
        if s in valSmilesSet:
            valSmiles.append(s)
            valProtein.append(datasetProtein[i])
            valLabel.append(datasetLabel[i])
        if s in testSmilesSet:
            testSmiles.append(s)
            testProtein.append(datasetProtein[i])
            testLabel.append(datasetLabel[i])

    return trainSmiles, trainProtein, trainLabel, valSmiles, valProtein, valLabel, testSmiles, testProtein, testLabel
