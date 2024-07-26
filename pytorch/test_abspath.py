import os


path = "data/sda/asdas/asd/asd/asd"
print(path)
save_mask_dir = os.path.abspath(path)
print(save_mask_dir)




print(os.getcwd())

if save_mask_dir.startswith(u"\\\\"):
    save_mask_dir = u"\\\\?\\UNC\\" + save_mask_dir[2:]
else:
    save_mask_dir = u"\\\\?\\" + save_mask_dir
    
save_mask_dir_short = save_mask_dir.replace(u"\\\\?\\"+os.getcwd()+"\\", "")

print(save_mask_dir_short)