import xml.etree.ElementTree as ET
import numpy as np 
import pandas as pd 
from scipy.io import savemat
import os

def xml2mat(xmlfile):
	tree = ET.parse(xmlfile)
	root = tree.getroot()
	x = np.array([])
	y = np.array([])
	# z = np.array([])
	obj = np.array([])
	label = np.array([])
	for Annotation in root.iter('Annotation'):
		# print('cell type ID',Annotation.get('Id')) #cell type ID
		for Region in Annotation.iter('Region'):
			xx = np.array([int(Vertex.get('X')) for Vertex in Region.iter('Vertex')])
			yy = np.array([int(Vertex.get('Y')) for Vertex in Region.iter('Vertex')])
			# zz = np.array([int(Vertex.get('Z')) for Vertex in Region.iter('Vertex')])
			objj = np.array([int(Region.get('Id'))]*len(xx))
			labell = np.array([int(Annotation.get('Id'))]*len(xx))
			x = np.concatenate((x,xx),axis=None)
			y = np.concatenate((y,yy),axis=None)
			# z = np.concatenate((z,zz),axis=None)
			obj = np.concatenate((obj,objj),axis=None)
			label = np.concatenate((label,labell),axis=None)
	print('number of coordinates in annotation : ',len(x))
	mdict = {'x':x,'y':y,'objID':obj,'label':label}
	return mdict
# input
xmlfile = r'C:\Users\kuki\Desktop\Research\Skin\SVS\scan 2\Wirtz.Denis_OTS-19_5021-017.xml'
print(xmlfile)
# LUT for rename
# renameLUT = r'C:\Users\kuki\Desktop\Research\Skin\Datasheet\Young vs Old v3.xlsx'
# renameLUT = pd.read_excel(renameLUT, header=2)
# imagename = os.path.basename(xmlfile)
# imagename,extention = os.path.splitext(imagename)
# imageID = renameLUT['Image ID'][renameLUT['Proscia Image Name'] == imagename].values[0].astype('str')
# imageID = imageID.zfill(3) #fill zero in front of a number to make it three digits
# convert xml to mat and save 
mdict = xml2mat(xmlfile)
# output
outdir = os.path.dirname(xmlfile)
print('output file name is : ', 'Wirtz.Denis_OTS-19_5021-017.mat')
savemat(os.path.join(outdir,'Wirtz.Denis_OTS-19_5021-017.mat'),mdict=mdict,do_compression=True)



# xmldir = r'\\babyserverdw4\Pei-Hsun Wu\digital pathology image data\Skin Tissue - Kyu\SVS\scan 2'
# for xml in os.listdir(xmldir):
# 	if xml.endswith('.xml'):
# 		filename = os.path.splitext(xml)[0]
# 		mdict = xml2mat(os.path.join(xmlfile,xml))
# 		dst = renameLUT['New name'][renameLUT['Name on Proscia']==filename].values[0]
# 		savemat(os.path.join(outdir,dst+'.mat'),mdict=mdict,do_compression=True)