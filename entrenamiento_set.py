import pandas as pd
from datasets import load_dataset

if __name__ == '__main__':

    """ 
    training = pd.read_csv('training.csv',sep=';')
    X_train, X_test, y_train, y_test = train_test_split(training.Sentence, training.Metaphor, test_size=0.2, random_state=0)
    print(type(X_train))
    training_set=pd.concat([X_train.reset_index(drop=True),y_train.reset_index(drop=True)],axis=1)
    print(training_set)
    training_set.to_csv('training80.csv',sep=',',index=False)
    training.to_csv('training_total.csv',sep=',',index=False)
    """
    """
    training_set_l=[X_train,y_train]
    training_set=pd.DataFrame(training_set_l)
    print(training_set)
    training.to_csv('training_total.csv')
    training_set.to_csv('training80.csv')

    print(X_test, y_test)
    """


    prueba = load_dataset("csv", data_files="training_total.csv",split= None)
    print(prueba)
    print(prueba["train"][0])