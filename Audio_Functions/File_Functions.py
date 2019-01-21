import os
import pickle

def create_folders(repo_path):
    emotions=["positive","neutral","negative"]
    os.chdir(repo_path)
    if (os.path.exists("audio/test/"))==False:
        os.mkdir("audio/test")
        os.chmod("audio/test/",511)
    if (os.path.exists("audio/train/"))==False:
        os.mkdir("audio/train")
        os.chmod("audio/train/",511)
    for em_dir in emotions:
        if (os.path.exists("audio/test/"+em_dir))==False:
            #python understands octal so 0777 has to be 511 in decimal
            os.mkdir("audio/test/"+em_dir)
            os.chmod("audio/test/"+em_dir,511)
            print('Created directory audio/test/'+em_dir)
        else:
            print('Directory audio/test/'+em_dir+' exists.')

        if (os.path.exists("audio/train/"+em_dir))==False:
            #python understands octal so 0777 has to be 511 in decimal
            os.mkdir("audio/train/"+em_dir)
            os.chmod("audio/train/"+em_dir,511)
            print('Created directory audio/train/'+em_dir)
        else:
            print('Directory audio/train/'+em_dir+' exists.')

def create_folders_perID(repo_path,id,case):
    emotions=["positive","neutral","negative"]
    os.chdir(repo_path)
    print(id)
    if (os.path.exists("audio/"+id+"/"))==False:
        os.mkdir("audio/"+id+"/")
        os.chmod("audio/"+id+"/",511)
    if case is 'train':
        for em_dir in emotions:
            if (os.path.exists("audio/"+id+"/"+em_dir))==False:
                #python understands octal so 0777 has to be 511 in decimal
                os.mkdir("audio/"+id+"/"+em_dir)
                os.chmod("audio/"+id+"/"+em_dir,511)
                print('Created directory audio/' +id+ '/' +em_dir)
            else:
                print('Directory audio/' +id+ '/' +em_dir+' exists.')
        

def remove_folders(repo_path):
    train = repo_path + "/audio/train"
    test = repo_path + "/audio/test"
    print(test,train)
    for root, dirs, files in os.walk(train, topdown=False):
        for name in files:
            #print(os.path.join(root, name)) 
            os.remove(os.path.join(root, name))
        for name in dirs:
            #print(os.path.join(root, name))
            os.rmdir(os.path.join(root, name))
           

    for root, dirs, files in os.walk(test, topdown=False):
        for name in files:
            #print(os.path.join(root, name))
            os.remove(os.path.join(root, name))
            
        for name in dirs:
            #print(os.path.join(root, name))
            os.rmdir(os.path.join(root, name))

def retrieveSubs(subsPath,repo_path):
    os.chdir(repo_path)
    subtitles_pol_file=open(subsPath, 'rb')
    # Loading the Subtitle
    subtitles_pol = pickle.load(subtitles_pol_file)
    return subtitles_pol





