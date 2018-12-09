import cv2
import numpy as np

embeddingfile = "../data/glove.6B/glove.6B.50d.txt"
emb_dim = 50
GloveEmbeddings = {}
query_len = 12
passage_len = 50

def loadEmbeddings(embeddingfile):
    fe = open(embeddingfile,"r")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        vec = tokens[1:]
        vec = " ".join(vec)
        float_vec = []
        for num in vec.split(' '):
        	float_vec.append(float(num))
        GloveEmbeddings[word]=float_vec
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddings["zerovec"] = [0.0001 for i in range(0, emb_dim)]
    fe.close()


def convert_input_to_vector(sentence, length):	
	ret = []
	words = sentence.split()
	count = 0
	for word in words:
		try:
			ret.append(GloveEmbeddings[word])
			count += 1
		except:
			pass
		if(count >= length):
			break

	while(count != length):
		ret.append(GloveEmbeddings["zerovec"])
		count += 1

	return ret


def return_vectors():
	loadEmbeddings(embeddingfile)
	file = open("../data/one_perc_data.tsv")
	lines = file.read().split('\n')
	ret_label = []
	ret_query_vec = []
	ret_para_vec = []
	for i in range(0, len(lines) - 1, 10):
		row = lines[i].split('\t')
		query = row[1].lower()
		query_vec = convert_input_to_vector(query, query_len)
		para_vec = []
		label = -1
		for j in range(0, 10):
			if(int(lines[i + j].split('\t')[3]) == 1):
				label = j
			para = lines[i + j].split('\t')[2]
			para_vec.append(convert_input_to_vector(para, passage_len))
		# print label
		label = [(1 if i == label else 0) for i in range(0, 10)]
		ret_label.append(label)
		ret_para_vec.append(para_vec)
		ret_query_vec.append(query_vec)

	ret_label = np.array(ret_label)
	ret_query_vec = np.array(ret_query_vec, dtype = np.float64)
	ret_para_vec = np.array(ret_para_vec, dtype = np.float64)

	return ret_label, ret_query_vec, ret_para_vec

if __name__ == "__main__":
	l, q, p = return_vectors()
	print l.shape, q.shape, p.shape
