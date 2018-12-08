import math
import re
import fileinput
from PorterStemmer import PorterStemmer

KEYWORDS_PATH = "keywords-lab3.txt"
DOCUMENTS_PATH = "documents-lab3.txt"
MAX_PRINT_RESULTS = 10


class Stemmer:
    @staticmethod
    def stem(token):
        # use PorterStemmer to stem token
        p = PorterStemmer()
        token = p.stem(token, 0, len(token) - 1)

        return token


class Dictionary:
    def __init__(self, keywords):
        self._terms = list(set([Stemmer.stem(x) for x in keywords]))
        self._idfs = {}

    def calculateIdfs(self, documents):
        # calculate idfs for each term - log(N / m) - N - documents count, m - number of documents containing given
        # term assign computed values to _idfs map (key: term, value: IDF)

        lDocumentNumber = len(documents)
        lTermCount = {}
        for lDocument in documents:
            for term in lDocument._terms:
                if term not in lTermCount:
                    lTermCount[term] = 0
                    for lInnerDocument in documents:
                        if term in lInnerDocument._terms:
                            lTermCount[term] += 1
                    self._idfs[term] = math.log(lDocumentNumber/lTermCount[term])

        # print("Calculated idfs: ")
        # for lTerm, lIdf in self._idfs.items():
        #     print("\t(" + lTerm + " : " + str(lIdf) + ")")

        pass

    def getTerms(self):
        return self._terms

    def getIfs(self):
        return self._idfs

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

        #  1.remove non-alphanumeric signs, keep only letters, digits and spaces.
        content = re.sub(r'[^a-zA-Z0-9 ]', "", content)

        #  2. remove multiple spaces in a row if exist
        content = re.sub(' +', ' ', content)

        #  3.change text to lowercase
        content = content.lower();

        return content

    def tokenizeDocument(self, normalized):
        # tokenize document - use simple division on white spaces.
        normalized = normalized.split(' ');

        return normalized

    def calculateBagOfWords(self, terms, dictionary):
        #  calculate bag-of-words representation - count how many times each term from dictionary.getTerms
        #  exists in document

        lBagOfWordsDictionary = {}
        for term in dictionary.getTerms():
            if term not in lBagOfWordsDictionary:
                lCount = terms.count(term)
                lBagOfWordsDictionary[term] = lCount

        return lBagOfWordsDictionary

    def calculateTfs(self, bagOfWords):
        # calculate TF representation - divide elements from bag-of-words by maximal value from this vector

        lTFDictionary = {}
        for lWord in bagOfWords:
            if lWord not in lTFDictionary:
                if max(bagOfWords.values()) != 0:
                    lTFDictionary[lWord] = bagOfWords[lWord] / max(bagOfWords.values())
                else:
                    lTFDictionary[lWord] = 0

        return lTFDictionary

    def calculateTfIds(self, tfs, dictionary):
        # calculate TF-IDF representation - multiply elements from tf representation my matching IDFs (dictionary.getIfs())
        # return results as list of tf-IDF values for terms in the same order as dictionary.getTerms()

        lTfIdsList = []
        for term in dictionary.getTerms():
            lTfIdsList.append(tfs[term] * dictionary.getIfs()[term])

        return lTfIdsList

    def calculateSimilarity(self, query):
        # TODO: calculate cosine similarity between current document and query document
        # (use calculated TF_IDFs - _tfidfs field)
        # similarity = sum(a[i]*q[i])/(sqrt(sum(a[i]^2)*sum(q[i]^2)))
        # if denominator is equal to 0 return 0 to prevent division by 0

        lSumAQ = 0
        lSumA = 0
        lSumQ = 0
        lSimilarity = 0

        for i in range(len(self._tfIdfs)):
            lSumAQ += self._tfIdfs[i] * query._tfIdfs[i]
            lSumA += self._tfIdfs[i] * self._tfIdfs[i]
            lSumQ += query._tfIdfs[i] * query._tfIdfs[i]

        lDenominator = math.sqrt(lSumA * lSumQ)
        if lDenominator != 0:
            lSimilarity = lSumAQ / lDenominator

        return lSimilarity


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
