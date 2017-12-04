import os,sys
from random import random
dir_path = sys.argv[1]
tr_files_path = os.path.join(dir_path,'training/myimage_2')
files = os.listdir(tr_files_path)
for curr_file in files:
	if(random() > 0.20):
		os.system("cp "+ dir_path +"/training/myimage_2/"+curr_file +" "+dir_path+"/split_train/myimage_2/")
		os.system("cp "+ dir_path +"/training/myimage_3/"+curr_file +" "+dir_path+"/split_train/myimage_3/")
		os.system("cp "+ dir_path +"/training/disp_noc_0/"+curr_file +" "+dir_path+"/split_train/disp_noc_0/")
		os.system("cp "+ dir_path +"/training/disp_occ_0/"+curr_file +" "+dir_path+"/split_train/disp_occ_0/")
	else:
		os.system("cp "+ dir_path +"/training/myimage_2/"+curr_file +" "+dir_path+"/split_val/myimage_2/")
		os.system("cp "+ dir_path +"/training/myimage_3/"+curr_file +" "+dir_path+"/split_val/myimage_3/")
		os.system("cp "+ dir_path +"/training/disp_noc_0/"+curr_file +" "+dir_path+"/split_val/disp_noc_0/")
		os.system("cp "+ dir_path +"/training/disp_occ_0/"+curr_file +" "+dir_path+"/split_val/disp_occ_0/")