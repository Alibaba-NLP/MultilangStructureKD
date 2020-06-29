import pdb
import os
import subprocess
tar_dir='sub10'
files=os.listdir(tar_dir)
# write=False
# # target='test'
target='conllu'

for file in files:
	# pdb.set_trace()
	filedir=os.path.join(tar_dir,file)
	# if not os.path.isdir(filedir):
	# 	continue
	# if 'UD_' not in file:
	# 	continue
	# print(file,end=' ')
	print(file)
	filename=filedir
	if target in filename and 'conll' in filename:
		# print(filename)
		# count_projective(os.path.join(filedir,filename))
		# count_additional(os.path.join(filedir,filename))	
		# pdb.set_trace()
		names=filename.split('.')
		names[-2]=names[-2]+'_collapsed'
		# pdb.set_trace()
		with open('.'.join(names),'w') as outfile:
			subprocess.run(['perl','tools/enhanced_collapse_empty_nodes.pl',filename],stdout=outfile)
	continue

	# pdb.set_trace()
	for filename in os.listdir(filedir):
		
		# if target not in filename:# or '_modified' not in filename:
		# 	continue
		# pdb.set_trace()
		if target in filename and 'conll' in filename:
			# print(filename)
			# count_projective(os.path.join(filedir,filename))
			# count_additional(os.path.join(filedir,filename))	
			# pdb.set_trace()
			names=filename.split('.')
			names[-2]=names[-2]+'_collapsed'

			with open(os.path.join(filedir,'.'.join(names)),'w') as outfile:
				subprocess.run(['perl','tools/enhanced_collapse_empty_nodes.pl',os.path.join(filedir,filename)],stdout=outfile)
		# break
	# sets=os.listdir()
	# pdb.set_trace()
	# for dataset in sets:
	#   if '_modified' not in dataset:
	#       continue
	#   preprocessing(os.path.join(tar_dir,file,dataset))
	# pdb.set_trace()
