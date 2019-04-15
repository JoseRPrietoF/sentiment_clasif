import os, sys


def create_structure(dir_name):
    """
    Make a structure
    :param structure:
    :return:
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    if not os.path.exists(dir_name+"/checkpoints"):
        os.makedirs(dir_name+"/checkpoints")

    if not os.path.exists(dir_name+"/checkpoints/best"):
        os.makedirs(dir_name+"/checkpoints/best")

    if not os.path.exists(dir_name+"/output"):
        os.makedirs(dir_name+"/output")



def write_from_array(array, path):
    """
    Aux method to call write_output from an array of results
    :return:
    """
    for a in array:
        id, et = a
        write_output(path=path, id_tweet=id, etiqueta=et) #set        path+'/'+lang

def write_output(path, id_tweet, etiqueta): #set
    """
    <author id="author-id" lang="en|es" type="bot|human" gender="bot|male|female" />
    gender not predicted at the moment
    :param id:
    :return:
    """
    str = '{} \t {}\n'.format(id_tweet, etiqueta) #set
    with open(path, 'a') as the_file:
        the_file.write(str)
        the_file.close()