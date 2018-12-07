from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import cv2

embeddingfile = "data/glove.6B/glove.6B.300d.txt"
emb_dim = 300
GloveEmbeddings = {}

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
    GloveEmbeddings["zerovec"] = [0.0 for i in range(0, emb_dim)]
    fe.close()


def similarity(X, Y):
	X = list(X)
	Y = list(Y)
	return cosine_similarity(X, Y)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



def load_test_file(in_file_name, out_file_name):
	file = open(in_file_name)
	lines = file.read().split('\n')
	file_out = open(out_file_name, 'w')

	exceptions = ['.', ',', '-', '?', '!', ':', ';', '\'', ')']
	qid_array = []

	print "Total Queries : ", len(lines)/10
	cv2.waitKey(0)

	for i in range(0, len(lines) - 1, 10):
		print "Query number ", i/10, " Total : ", len(lines)/10
		qid = lines[i].split('\t')[0]
		qid_array.append(qid)
		query = lines[i].split('\t')[1]
		query = query.lower()
		query_vector = [0 for k in range(0, emb_dim)]
		query_vector = np.array(query_vector, dtype = np.float64)
		count = 0
		for word in query.split(' '):
			if(word == ''):
				continue
			if(len(word) == 1):
				continue
			if(word[len(word) - 1] in exceptions):
				word = word[:len(word)-1]

			try:
				query_vector += GloveEmbeddings[word]
			except:
				print "except", word
				query_vector += GloveEmbeddings["zerovec"]
			count += 1
		if(count != 0):
			query_vector /= count


		para_vector_array = []
		for j in range(0, 10):
			count = 0
			para_vector = [0 for k in range(0, emb_dim)]
			para_vector = np.array(para_vector, dtype = np.float64)

			para = lines[i + j].split('\t')[2]
			para = para.lower()
			for word in para.split(' '):
				if(word == ''):
					continue
				if(len(word) == 1):
					continue
				if(word[len(word) - 1] in exceptions):
					word = word[:len(word)-1]
				try:
					para_vector += GloveEmbeddings[word]
				except:
					print "para except", word
					para_vector += GloveEmbeddings["zerovec"]
				count += 1
			if(count != 0):
				para_vector /= count
			para_vector_array.append(para_vector)

		prob = [0 for k in range(0,10)]

		for k in range(0, 10):
			X = np.expand_dims(query_vector, 0)
			Y = np.expand_dims(para_vector_array[k], 0)
			prob[k] = similarity(X, Y)

		prob = np.array(prob)
		prob = np.squeeze(prob)
		prob = softmax(prob)
		# print qid, prob
		file_out.write(str(qid))
		file_out.write('\t')
		for k in range(0, 10):
			file_out.write(str(prob[k]))
			if(k != 9):
				file_out.write('\t')
		file_out.write('\n')

loadEmbeddings(embeddingfile)
load_test_file("data/eval1_unlabelled.tsv", "submit.tsv")

# prob = [0 for i in range(0,10)]
# for i in range(0, 10):
# 	X = np.expand_dims(query_vector, 0)
# 	Y = np.expand_dims(para_vector[i], 0)
# 	prob[i] = similarity(X, Y)

# prob = np.array(prob)
# prob = np.squeeze(prob)
# prob = softmax(prob)
# print prob
# print qid