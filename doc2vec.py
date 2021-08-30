from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk


class Doc_to_vec :
    def __init__(self, document, epochs=100, embedding_size=64, alpha=0.01, min_alpha=0.00025, min_count=1, dm=1, model_name=None) :
        # data
        self.document = document
        #self.tagged_data = self.text_preprocessing()

        # parameters
        self.epochs= int(epochs)
        self.embedding_size = int(embedding_size)
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = int(min_count)
        self.dm = int(dm)
        self.model_name = model_name

    def text_preprocessing(self) :
        '''
        Do not use this method 
        '''
        return [TaggedDocument(words=nltk.word_tokenize(word.lower()), tags=[str(index)]) for index, word in enumerate(self.document)]

    def train(self) :
        self.doc2vec_model = Doc2Vec(vector_size=self.embedding_size, alpha=self.alpha, min_alpha=self.min_alpha, min_count=self.min_count, dm=self.dm)
        self.doc2vec_model.build_vocab(self.document)

        for epoch in range(1, self.epochs+1) :
            if epoch % 10 == 0 :
                print('Epochs : {}'.format(epoch))
            self.doc2vec_model.train(self.document, 
                                total_examples=self.doc2vec_model.corpus_count,
                                epochs=self.doc2vec_model.epochs)
            self.doc2vec_model.alpha -= 0.0002
            
        self.doc2vec_model.save(self.model_name)
        print("Doc2vec model saved")





