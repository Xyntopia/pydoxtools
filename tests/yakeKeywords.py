import yake

text = "Sources tell us that Google is acquiring Kaggle, a platform that hosts data science and machine learning " \
       "competitions. Details about the transaction remain somewhat vague, but given that Google is hosting its Cloud " \
       "Next conference in San Francisco this week, the official announcement could come as early as tomorrow. " \
       "Reached by phone, Kaggle co-founder CEO Anthony Goldbloom declined to deny that the acquisition is happening. " \
       "Google itself declined 'to comment on rumors'. Kaggle, which has about half a million data scientists on its platform, " \
       "was founded by Goldbloom  and Ben Hamner in 2010. " \
       "The service got an early start and even though it has a few competitors like DrivenData, TopCoder and HackerRank, " \
       "it has managed to stay well ahead of them by focusing on its specific niche. " \
       "The service is basically the de facto home for running data science and machine learning competitions. " \
       "With Kaggle, Google is buying one of the largest and most active communities for data scientists - and with that, " \
       "it will get increased mindshare in this community, too (though it already has plenty of that thanks to Tensorflow " \
       "and other projects). Kaggle has a bit of a history with Google, too, but that's pretty recent. Earlier this month, " \
       "Google and Kaggle teamed up to host a $100,000 machine learning competition around classifying YouTube videos. " \
       "That competition had some deep integrations with the Google Cloud Platform, too. Our understanding is that Google " \
       "will keep the service running - likely under its current name. While the acquisition is probably more about " \
       "Kaggle's community than technology, Kaggle did build some interesting tools for hosting its competition " \
       "and 'kernels', too. On Kaggle, kernels are basically the source code for analyzing data sets and developers can " \
       "share this code on the platform (the company previously called them 'scripts'). " \
       "Like similar competition-centric sites, Kaggle also runs a job board, too. It's unclear what Google will do with " \
       "that part of the service. According to Crunchbase, Kaggle raised $12.5 million (though PitchBook says it's $12.75) " \
       "since its   launch in 2010. Investors in Kaggle include Index Ventures, SV Angel, Max Levchin, Naval Ravikant, " \
       "Google chief economist Hal Varian, Khosla Ventures and Yuri Milner "

text = ("Quellen berichten uns, dass Google Kaggle übernimmt, eine Plattform, die Wettbewerbe "
        "in den Bereichen Datenwissenschaft und maschinelles Lernen veranstaltet. Einzelheiten "
        "über die Transaktion bleiben etwas vage, aber da Google diese Woche seine Cloud Next-"
        "Konferenz in San Francisco ausrichtet, könnte die offizielle Ankündigung bereits morgen "
        "erfolgen. Per Telefon erreicht, lehnte Kaggle-Mitbegründer CEO Anthony Goldbloom es ab, "
        "zu leugnen, dass die Übernahme stattfindet. Google selbst lehnte es ab, 'zu Gerüchten"
        " Stellung zu nehmen'. Kaggle, das etwa eine halbe Million Datenwissenschaftler auf seiner"
        " Plattform hat, wurde 2010 von Goldbloom und Ben Hamner gegründet. Der Dienst hatte einen "
        "frühen Start und obwohl er einige Konkurrenten wie DrivenData, TopCoder und HackerRank hat, "
        "ist es ihm gelungen, durch die Konzentration auf seine spezifische Nische weit vor ihnen "
        "zu bleiben. Der Dienst ist im Grunde das de facto Zuhause für die Durchführung von Wettbewerben "
        "in den Bereichen Datenwissenschaft und maschinelles Lernen. Mit Kaggle kauft Google eine "
        "der größten und aktivsten Gemeinschaften für Datenwissenschaftler - und damit erhöht sich "
        "auch sein Einfluss in dieser Gemeinschaft (obwohl es dank Tensorflow und anderen Projekten "
        "bereits viel davon hat). Kaggle hat auch eine gewisse Geschichte mit Google, aber das ist "
        "ziemlich neu. Anfang dieses Monats haben sich Google und Kaggle zusammengetan, um einen "
        "100.000-Dollar-Wettbewerb im Bereich maschinelles Lernen zur Klassifizierung von YouTube-"
        "Videos zu veranstalten. Dieser Wettbewerb hatte auch tiefe Integrationen mit der Google "
        "Cloud Platform. Unseres Erachtens wird Google den Dienst weiterführen - wahrscheinlich "
        "unter seinem aktuellen Namen. Während die Übernahme wahrscheinlich mehr mit Kaggle's "
        "Gemeinschaft als mit der Technologie zu tun hat, hat Kaggle einige interessante Werkzeuge"
        " zum Hosting seiner Wettbewerbe und 'Kernels' entwickelt. Auf Kaggle sind Kerne im Grunde"
        " der Quellcode für die Analyse von Datensätzen, und Entwickler können diesen Code auf der "
        "Plattform teilen (das Unternehmen nannte sie früher 'Skripte'). Wie ähnliche "
        "wettbewerbszentrierte Websites betreibt Kaggle auch eine Jobbörse. Es ist unklar, "
        "was Google mit diesem Teil des Dienstes tun wird. Laut Crunchbase hat Kaggle seit "
        "seinem Start im Jahr 2010 12,5 Millionen Dollar eingesammelt (obwohl PitchBook sagt, "
        "es sind 12,75), Investoren in Kaggle umfassen Index Ventures, SV Angel, Max Levchin, "
        "Naval Ravikant, Google-Chefökonom Hal Varian, Khosla Ventures und Yuri Milner.")

"""
kw_extractor = yake.KeywordExtractor()
keywords = kw_extractor.extract_keywords(text)

for kw in keywords:
	print(kw)
"""

language = "en"
max_ngram_size = 3
deduplication_threshold = 0.9
deduplication_algo = 'seqm'
windowSize = 1
numOfKeywords = 20

custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                            dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords,
                                            features=None)
keywords = custom_kw_extractor.extract_keywords(text)

for kw in keywords:
    print(kw)
