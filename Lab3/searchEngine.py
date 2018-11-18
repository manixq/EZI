import math
import re
import fileinput
from PorterStemmer import PorterStemmer

KEYWORDS_PATH = "keywords.txt"
DOCUMENTS_PATH = "documents.txt"
MAX_PRINT_RESULTS = 10


class Stemmer:
    @staticmethod
    def stem(token):
        # TODO: use PorterStemmer to stem token
        # you can see exemplary use of stemmer in method main of PorterStemmer class
        return token


class Dictionary:
    def __init__(self, keywords):
        self._terms = list(set([Stemmer.stem(x) for x in keywords]))
        self._idfs = {}

    def calculateIdfs(self, documents):
        # TODO: calculate idfs for each term - log(N / m) - N - documents count, m - number of documents containing given
        # term assign computed values to _idfs map (key: term, value: IDF)
        pass

class Document:
    def __init__(self, content, title):
        self._content = content
        self._title = title
        self._terms = []
        self._tfIdfs = []
        self.preprocessDocument()

    def preprocessDocument(self):
        normalized = self.normalizeText(self._content)
        tokens = self.tokenizeDocument(normalized)
        self._terms = self.stemTokens(tokens)

    def calculateRepresentations(self, dictionary):
        bagOfWords = self.calculateBagOfWords(self._terms, dictionary)
        tfs = self.calculateTfs(bagOfWords)
        self._tfIdfs = self.calculateTfIds(tfs, dictionary)

    def stemTokens(self, tokens):
        return [Stemmer.stem(x) for x in tokens]

    def normalizeText(self, content):
        # TODO:
        #  1.remove non-alphanumeric signs, keep only letters, digits and spaces.
        #  2. remove multiple spaces in a row if exist
        #  3.change text to lowercase
        return content

    def tokenizeDocument(self, normalized):
        # TODO: tokenize document - use simple division on white spaces.
        return []

    def calculateBagOfWords(self, terms, dictionary):
        # TODO: calculate bag-of-words representation - count how many times each term from dictionary.getTerms
        #  exists in document
        return {}

    def calculateTfs(self, bagOfWords):
        # TODO: calculate TF representation - divide elements from bag-of-words by maximal value from this vector
        return {}

    def calculateTfIds(self, tfs, dictionary):
        # TODO: calculate TF-IDF representation - multiply elements from tf representation my matching IDFs (dictionary.getIfs())
        # return results as list of tf-IDF values for terms in the same order as dictionary.getTerms()
        return []

    def calculateSimilarity(self, query):
        # TODO: calculate cosine similarity between current document and query document
        # (use calculated TF_IDFs - _tfidfs field)
        # similarity = sum(a[i]*q[i])/(sqrt(sum(a[i]^2)*sum(q[i]^2)))
        # if denominator is equal to 0 return 0 to prevent division by 0
        return 0


#DONE: do not modify
class SearchEngine:
    def __init__(self):
        self._documents = None
        self._dictionary = None

    def run(self):
        self.loadDocuments()
        self.loadDictionary()
        self._dictionary.calculateIdfs(self._documents)
        for doc in self._documents:
            doc.calculateRepresentations(self._dictionary)
        while True:
            query = input()
            if query == "q":
                break
            queryDocument = Document(query, "query")
            queryDocument.calculateRepresentations(self._dictionary)
            similarities = {}
            for doc in self._documents:
                similarities[doc] = doc.calculateSimilarity(queryDocument)
            sortedSimilarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            for i in range(MAX_PRINT_RESULTS):
                if i >= len(self._documents):
                    break
                print(sortedSimilarities[i][0]._title + " " + str(sortedSimilarities[i][1]))

    def loadDictionary(self):
        keywords = []
        for keyword in fileinput.input(KEYWORDS_PATH):
            keywords.append(re.sub("\n", "", keyword.lower()))

        fileinput.close()
        self._dictionary = Dictionary(keywords)

    def loadDocuments(self):
        self._documents = []
        currentDocument = ""
        currentTitle = None
        for line in fileinput.input(DOCUMENTS_PATH):
            if "\n" == line:
                self._documents.append(Document(currentDocument.lstrip(), currentTitle))
                currentTitle = None
                currentDocument = ""
            else:
                if currentTitle is None:
                    currentTitle = re.sub("\n", "", line)
                currentDocument += " " + re.sub("\n", " ", line)
        if "" != currentDocument.lstrip():
            self._documents.append(Document(currentDocument.lstrip(), currentTitle))
        fileinput.close()


if __name__ == '__main__':
    engine = SearchEngine()
    engine.run()
