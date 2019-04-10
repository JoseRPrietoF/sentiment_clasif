from xml.dom import minidom

def get_tweets(path, test = False):
    X, y = [], []
    xmldoc = minidom.parse(path)
    tweets = xmldoc.getElementsByTagName('tweet')
    # outfile = open(path+'training.txt', 'w')
    for t in tweets:
        for node in t.childNodes:
            # print (t.childNodes)
            if node.nodeName == 'tweetid':
                id_t = node.firstChild.data
            elif node.nodeName == 'content':
                text_t = node.firstChild.data.replace("\n", " ")
            elif node.nodeName == 'sentiment':
                if not test:
                    for h in node.childNodes:
                        if h.nodeName == 'polarity':
                            if h.firstChild.nodeName == 'value':
                                pol_t = h.firstChild.firstChild.data
        # print(id_t, pol_t, text_t)
        # print(text_t)
        X.append(text_t)
        if not test:
            y.append(pol_t)
    if not test:
        return X, y
    else:
        return X

