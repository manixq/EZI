# Authors: 
# Michal Tomczyk
# Pawel Mania
# -------------------------------------------------------------------------
import urllib.request as req
import sys
import os
import numpy
from html.parser import HTMLParser


# -------------------------------------------------------------------------
### generatePolicy classes


# Dummy fetch policy. Returns first element. Does nothing ;)
class Dummy_Policy:
    def getURL(self, c, iteration):
        if len(c.URLs) == 0:
            # instead of none
            c.URLs = list(c.seedURLs)
            return c.seedURLs[-1]
        else:
            return c.seedURLs[-1]

    def updateURLs(self, c, newURLs, newURLsWD, iteration):
        pass


class LIFO_Policy:
    def getURL(self, c, iteration):
        if len(c.queue) == 0:
            c.queue = list(c.seedURLs)
            temp = c.queue[-1]
            del c.queue[-1]
            return temp
        else:
            temp = c.queue[-1]
            del c.queue[-1]
            return temp

    def updateURLs(self, c, newURLs, newURLsWD, iteration):
        tmpList = list(newURLs)
        tmpList.sort(key=lambda url: url[len(url) - url[::-1].index('/'):])
        c.queue.extend(tmpList)


class FIFO_Policy:
    def getURL(self, c, iteration):
        if len(c.queue) == 0:
            c.queue = list(c.seedURLs)
            temp = c.queue[0]
            del c.queue[0]
            return temp
        else:
            temp = c.queue[0]
            del c.queue[0]
            return temp

    def updateURLs(self, c, newURLs, newURLsWD, iteration):
        tmpList = list(newURLs)
        tmpList.sort(key=lambda url: url[len(url) - url[::-1].index('/'):])
        c.queue.extend(tmpList)


class LIFO_Cycle_Policy:
    def __init__(self):
        self.fetched = set([])

    def getURL(self, c, iteration):
        if len(c.queue) == 0:
            self.fetched.clear()
            c.queue = list(c.seedURLs)
            temp = c.queue[-1]
            del c.queue[-1]
            return temp
        else:
            temp = c.queue[-1]
            while (temp in self.fetched):
                if len(c.queue) > 0:
                    del c.queue[-1]
                    if len(c.queue) > 0:
                        temp = c.queue[-1]
                else:
                    self.fetched.clear()
                    c.queue = list(c.seedURLs)
                    temp = c.queue[-1]
                    del c.queue[-1]
                    return temp
            return temp

class LIFO_Authority_Policy:
    def __init__(self):
        self.fetched = set([])

    def getURL(self, c, iteration):
        if len(c.queue) == 0:
            self.fetched.clear()
            c.queue = list(c.seedURLs)

        AutohorityLevels = []
        AutohorityLevelsSum = 0

        for url in c.queue:
            AutohorityLevelsSum += (len(c.incomingURLs) + 1)
        for url in c.queue:
            AutohorityLevels.append((len(c.incomingURLs) + 1) / AutohorityLevelsSum)

        temp = numpy.random.choice(c.queue, p = AutohorityLevels)
        c.queue.remove(temp)
        return temp

    def updateURLs(self, c, newURLs, newURLsWD, iteration):
        tmpList = list(newURLs)
        tmpList.sort(key=lambda url: url[len(url) - url[::-1].index('/'):])
        c.queue.extend(tmpList)


# -------------------------------------------------------------------------
# Data container
class Container:
    def __init__(self):
        # The name of the crawler"
        self.crawlerName = "IRbot"
        # Example ID
        self.example = "exercise3"
        # Root (host) page
        self.rootPage = "http://www.cs.put.poznan.pl/alabijak/ezi/lab1/" + self.example
        # Initial links to visit
        self.seedURLs = ["http://www.cs.put.poznan.pl/alabijak/ezi/lab1/"
                         + self.example + "/s0.html"]
        self.queue = list(self.seedURLs)
        # Maintained URLs
        self.URLs = set([])
        # Outgoing URLs (from -> list of outgoing links)
        self.outgoingURLs = {}
        # Incoming URLs (to <- from; set of incoming links)
        self.incomingURLs = {}
        # Class which maintains a queue of urls to visit. 
        self.generatePolicy = LIFO_Authority_Policy()
        # Page (URL) to be fetched next
        self.toFetch = None
        # Number of iterations of a crawler. 
        self.iterations = 50

        # If true: store all crawled html pages in the provided directory.
        self.storePages = True
        self.storedPagesPath = "./" + self.example + "/pages/"
        # If true: store all discovered URLs (string) in the provided directory
        self.storeURLs = True
        self.storedURLsPath = "/" + self.example + "/urls/"
        # If true: store all discovered links (dictionary of sets: from->set to),
        # for web topology analysis, in the provided directory
        self.storeOutgoingURLs = True
        self.storedOutgoingURLs = "/" + self.example + "/outgoing/"
        # Analogously to outgoing
        self.storeIncomingURLs = True
        self.storedIncomingURLs = "/" + self.example + "/incoming/"

        # If True: debug
        self.debug = False


class Parser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.output_list = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.output_list.append(dict(attrs).get('href'))


def main():
    # Initialise data
    c = Container()
    # Inject: parse seed links into the base of maintained URLs
    inject(c)

    # Iterate...
    for iteration in range(c.iterations):

        if c.debug:
            print("=====================================================")
            print("Iteration = " + str(iteration + 1))
            print("=====================================================")
        # Prepare a next page to be fetched
        generate(c, iteration)
        if (c.toFetch == None):
            if c.debug:
                print("   No page to fetch!")
            continue

        # Generate: it downloads html page under "toFetch URL"
        page = fetch(c)

        if page == None:
            if c.debug:
                print("   Unexpected error; skipping this page")
            removeWrongURL(c)
            continue

        # Parse file
        htmlData, newURLs = parse(c, page, iteration)

        # Store pages
        if c.storePages:
            storePage(c, htmlData)

        ### normalise newURLs
        newURLs = getNormalisedURLs(newURLs)

        ### update outgoing/incoming links
        updateOutgoingURLs(c, newURLs)
        updateIncomingURLs(c, newURLs)

        ### Filter out some URLs
        newURLs = getFilteredURLs(c, newURLs)

        ### removeDuplicates
        newURLsWD = removeDuplicates(c, newURLs)

        ### update urls
        c.generatePolicy.updateURLs(c, newURLs, newURLsWD, iteration)

        # Add newly obtained URLs to the container   
        if c.debug:
            print("   Maintained URLs...")
            for url in c.URLs:
                print("      " + str(url))

        if c.debug:
            print("   Newly obtained URLs (duplicates with maintaines URLs possible) ...")
            for url in newURLs:
                print("      " + str(url))
        if c.debug:
            print("   Newly obtained URLs (without duplicates) ...")
            for url in newURLsWD:
                print("      " + str(url))
            for url in newURLsWD:
                c.URLs.add(url)

    # store urls
    if c.storeURLs:
        storeURLs(c)
    if c.storeOutgoingURLs:
        storeOutgoingURLs(c)
    if c.storeIncomingURLs:
        storeIncomingURLs(c)

    # -------------------------------------------------------------------------


# Inject seed URL into a queue (DONE)
def inject(c):
    for l in c.seedURLs:
        if c.debug:
            print("Injecting " + str(l))
        c.URLs.add(l)


# -------------------------------------------------------------------------
# Produce next URL to be fetched (DONE)
def generate(c, iteration):
    url = c.generatePolicy.getURL(c, iteration)
    if url == None:
        if c.debug:
            print("   Fetch: error")
        c.toFetch = None
        return None
    # WITH NO DEBUG!
    print("   Next page to be fetched = " + str(url))
    c.generatePolicy.fetched.add(url)
    c.toFetch = url


# -------------------------------------------------------------------------
# Generate (download html) page (DONE)
def fetch(c):
    URL = c.toFetch
    if c.debug:
        print("   Downloading " + str(URL))
    try:
        opener = req.build_opener()
        opener.addheadders = [('User-Agent', c.crawlerName)]
        webPage = opener.open(URL)
        return webPage
    except:
        return None

    # -------------------------------------------------------------------------


# Remove wrong URL (TODO)
def removeWrongURL(c):
    c.URLs.remove(c.toFetch)


# -------------------------------------------------------------------------
# Parse this page and retrieve text (whole page) and URLs (TODO)
def parse(c, page, iteration):
    # data to be saved (DONE)
    htmlData = page.read()
    # obtained URLs (TODO)
    newURLs = set([])
    p = Parser()
    p.feed(str(htmlData))

    newURLs.update(p.output_list)
    if c.debug:
        print("   Extracted " + str(len(newURLs)) + " links")
    return htmlData, newURLs


# -------------------------------------------------------------------------
# Normalise newly obtained links (TODO)
def getNormalisedURLs(newURLs):
    newURLs = [k.lower() for k in newURLs]
    return newURLs


# -------------------------------------------------------------------------
# Remove duplicates (duplicates) (TODO)
def removeDuplicates(c, newURLs):
    # TODO
    lNewURLs = set([])
    for lNewUrl in newURLs:
        if (lNewUrl not in c.URLs):
            lNewURLs.add(lNewUrl)
    return lNewURLs


# -------------------------------------------------------------------------
# Filter out some URLs (TODO)
def getFilteredURLs(c, newURLs):
    while c.toFetch in newURLs:
        newURLs.remove(c.toFetch)
    toLeft = set([url for url in newURLs if url.lower().startswith(c.rootPage)])
    if c.debug:
        print("   Filtered out " + str(len(newURLs) - len(toLeft)) + " urls")
    return toLeft


# -------------------------------------------------------------------------
# Store HTML pages (DONE)  
def storePage(c, htmlData):
    relBeginIndex = len(c.rootPage)
    totalPath = "./" + c.example + "/pages/" + c.toFetch[relBeginIndex + 1:]

    if c.debug:
        print("   Saving HTML page " + totalPath + "...")

    totalDir = os.path.dirname(totalPath)

    if not os.path.exists(totalDir):
        os.makedirs(totalDir)

    with open(totalPath, "wb+") as f:
        f.write(htmlData)
        f.close()


# -------------------------------------------------------------------------
# Store URLs (DONE)  
def storeURLs(c):
    relBeginIndex = len(c.rootPage)
    totalPath = "./" + c.example + "/urls/urls.txt"

    if c.debug:
        print("Saving URLs " + totalPath + "...")

    totalDir = os.path.dirname(totalPath)

    if not os.path.exists(totalDir):
        os.makedirs(totalDir)

    data = [url for url in c.URLs]
    data.sort()

    with open(totalPath, "w+") as f:
        for line in data:
            f.write(line + "\n")
        f.close()


# -------------------------------------------------------------------------
# Update outgoing links (DONE)  
def updateOutgoingURLs(c, newURLsWD):
    if c.toFetch not in c.outgoingURLs:
        c.outgoingURLs[c.toFetch] = set([])
    for url in newURLsWD:
        c.outgoingURLs[c.toFetch].add(url)


# -------------------------------------------------------------------------
# Update incoming links (DONE)  
def updateIncomingURLs(c, newURLsWD):
    for url in newURLsWD:
        if url not in c.incomingURLs:
            c.incomingURLs[url] = set([])
        c.incomingURLs[url].add(c.toFetch)


# -------------------------------------------------------------------------
# Store outgoing URLs (DONE)  
def storeOutgoingURLs(c):
    relBeginIndex = len(c.rootPage)
    totalPath = "./" + c.example + "/outgoing_urls/outgoing_urls.txt"

    if c.debug:
        print("Saving URLs " + totalPath + "...")

    totalDir = os.path.dirname(totalPath)

    if not os.path.exists(totalDir):
        os.makedirs(totalDir)

    data = [url for url in c.outgoingURLs]
    data.sort()

    with open(totalPath, "w+") as f:
        for line in data:
            s = list(c.outgoingURLs[line])
            s.sort()
            for l in s:
                f.write(line + " " + l + "\n")
        f.close()


# -------------------------------------------------------------------------
# Store incoming URLs (DONE)  
def storeIncomingURLs(c):
    relBeginIndex = len(c.rootPage)
    totalPath = "./" + c.example + "/incoming_urls/incoming_urls.txt"

    if c.debug:
        print("Saving URLs " + totalPath + "...")

    totalDir = os.path.dirname(totalPath)

    if not os.path.exists(totalDir):
        os.makedirs(totalDir)

    data = [url for url in c.incomingURLs]
    data.sort()

    with open(totalPath, "w+") as f:
        for line in data:
            s = list(c.incomingURLs[line])
            s.sort()
            for l in s:
                f.write(line + " " + l + "\n")
        f.close()


if __name__ == "__main__":
    main()
