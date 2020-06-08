import functools
import operator
from affine_equiv_ae import *
from utils import *
import time
import warnings
warnings.filterwarnings("ignore")

def main():
    dataset = "MNIST"
    n_train = 10000
    n_test = 1000
    batch_size = 128
    epochs = 500
    image_shape = (1,28,28); dim = functools.reduce(operator.mul, image_shape,1)

    train_gen = get_gen(dataset, 'train', batch_size, n=n_train)
    test_gen = get_gen(dataset,'test', batch_size, n=n_test)
    
    sigma = [0,0,0,0.8,0]
    model = AEAE(dim)
    params = model.parameters()
    print("Number of parameters = ",sum([np.prod(p.size()) for p in params]))

    t_list = []
    for e in range(epochs):
        print("Epoch",e)
        start = time.time()
        model = train_network(model, train_gen,n_train//batch_size,sigma,e)
        end = time.time()
        t_list.append(end-start)
    avg_time = sum(t_list)/len(t_list)
    print("Average time taken for one epoch =",str(avg_time))
    torch.save(model, 'models/AEAE_MNIST_final.th')

        # acc, _ = test_network(model, test_gen, n_test, n_test//batch_size,sigma)
if __name__ == "__main__":
    main()
