from affine_equiv_ae import *
from utils import *
import functools
import operator
import warnings
warnings.filterwarnings("ignore")

def find_ls(sc_em, reg_em):
    '''
    Calculate the estimated matrix representing the scale transformation (Ls)
    Input:
        sc_em = scaled images' embedding of dimension (number of images, D (dimension of embedding) ) 
        reg_em = original images' embedding of dimension (number of images, D ) 

    Output:
        numpy array  of dimension (D, D)
    '''
    return np.matmul(np.linalg.pinv(reg_em),sc_em)

def cal_equiv_err(sc_em,reg_em):
    '''
    Calculate the equivariance error on batch of images using the estimated Ls
    Input: 
        sc_em = scaled images' embedding of dimension (number of images, D (dimension of embedding) ) 
        reg_em = original images' embedding of dimension (number of images, D ) 
    Output:
        equivariance error (i.e. sum || Ls(f(x)) - f(Ls(x)) ||_2 )
    '''
    sc_em, reg_em = sc_em.cpu().detach(), reg_em.cpu().detach()
    ls = find_ls(sc_em,reg_em)
    ls = torch.Tensor(ls)
    tmp_em = torch.matmul(reg_em,ls)
    sum_ = 0
    for i in range(tmp_em.size(0)):
        sum_ += np.linalg.norm(tmp_em[i]-sc_em[i],2)/ np.linalg.norm(tmp_em[i])
    sum_ /= tmp_em.size(0)
    return ls, sum_

def equiv_test():
    dataset = "MNIST"
    n_train = 10000
    n_test = 1000
    batch_size = 128
    epochs = 500
    image_shape = (1,28,28); dim = functools.reduce(operator.mul, image_shape,1)

    train_gen = get_gen(dataset, 'train', batch_size, n=n_train)
    test_gen = get_gen(dataset,'test', batch_size, n=n_test)
    # affine transformation parameter    
    sigma = [0,0,0,0.8,0]

    acc_list = [] 
    ls_list = [] 
    model_dir = './models/AEAE_MNIST_final.th'
    model = torch.load(model_dir)
    params = model.parameters()
    print("Number of parameters = ",sum([np.prod(p.size()) for p in params]))

    test_gen = get_gen(
            "MNIST",'test', batch_size=batch_size,
            s=1.0, t=0.0,
            shuffle=False
    )
    #acc = test()
    s = 0
    for idx in range(n_test//batch_size):
        x, y = next(test_gen)
        x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
        _, _, _,_ , original_embedding, trans_embedding, trans_embedding_2 = model(x,sigma)
        ls, err = cal_equiv_err(trans_embedding,original_embedding)
        ls_list.append(ls)
        s+= err
    s /= n_test/batch_size
    print("\tEquivariance error of AEAE =",s)
    norm_stat = []
    for i,mat1 in enumerate(ls_list):
        for j, mat2 in enumerate(ls_list):
            if i > j:
                norm_stat.append(np.linalg.norm(mat1-mat2,2))
    print("\tStatistics of the estimated Ls values",np.mean(norm_stat), np.std(norm_stat)) 


if __name__ == "__main__":
    equiv_test()
