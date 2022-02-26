from sklearn import datasets
from regressiontree import TreeNode

def load_dataset(data = "wine"):
    
    if data == "wine":
        df = datasets.load_wine(as_frame = True)
    else: print('dataset not available')
    
    #save X and Y in the right format
    x = df.data
    y = df.target.values.tolist()

    return x,y

if __name__ == "__main__":

    X, Y = load_dataset(data = "wine")
    tree = TreeNode(X,Y)

    #grow the tree
    tree.grow_tree()
    #print the tree
    tree.print()

