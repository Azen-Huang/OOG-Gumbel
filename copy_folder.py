import os
import shutil
import sys

def copy_dir(iter):
    if(int(iter) > 0):
        source_folder = './c_model'
        destination_folder = './save_model/model' + str(iter)
        shutil.copytree(source_folder, destination_folder)
        print("copy to " + destination_folder)
    else:
    # fetch all files        
        destination_folder = './p_model/'
        source_folder = './c_model/'
        for file_name in os.listdir(source_folder):
            # construct full file path
            source = source_folder + file_name
            destination = destination_folder + file_name
            if os.path.isdir(source):
                #copy dir
                for sub_file_name in os.listdir(source):
                    sub_source = source + '/' + sub_file_name
                    sub_destination = destination + '/' + sub_file_name
                    if(os.path.isfile(sub_source)):
                        shutil.copy(sub_source, sub_destination)
                        #print('copied', sub_file_name)
            else:
                # copy only files
                if os.path.isfile(source):
                    shutil.copy(source, destination)
                    #print('copied', file_name)
        print("copy model complete!")
    
if __name__ == '__main__':
    iter = sys.argv[1]
    copy_dir(iter)