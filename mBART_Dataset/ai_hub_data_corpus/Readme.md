The dataset is based on the Korean-English AI Training Text Corpus (KEAT) provided by AI Open Innovation Hub, which is operated by the Korean National Information Society Agency (NIA).

The dataset is in a JSON format as follows :

[
	{ # example of the 1st article
        'catagory' : ['문화', '학술_문화재'],
        'date' : '2019-05-31T00:00:00.000Z',
        'source' : '국민일보',
        'text' :
        	[
                { # example of the 1st sentence in the 1st article
                'aihub_id' : 1530019,
                'en_ko_ner_gold' : [[0, 0], [2, 1]],
                'en_ner_auto' : [[2, 3, 'DATE'], [4, 8, 'ORG'], [22, 27, 'DATE']],
                'en_ner_gold' : [[2, 3, 'DATE'], [5, 8, 'ORG'], [22, 27, 'CARDINAL']],
                'en_text' : 'Founded in 1990, the Special Needs Institute has developed a variety of curative programs to study brain development and educate children under the age of 5 with developmental disabilities and autism.',
                'en_tokens' : ['Founded', 'in', '1990', ',', 'the', 'Special', 'Needs', 'Institute', 'has', 'developed', 'a', 'variety', 'of', 'curative', 'programs', 'to', 'study', 'brain', 'development', 'and', 'educate', 'children', 'under', 'the', 'age', 'of', '5', 'with', 'developmental', 'disabilities', 'and', 'autism', '.'], 
                'ko_ner_auto' : [[0, 2, 'DAT'], [14, 17, 'NOH']],
                'ko_ner_gold' : [[0, 2, 'DATE'], [14, 17, 'CARDINAL']],
                'ko_text' : '1990년 설립된 특수요육원은 뇌 발달을 연구해 만 5세 이하 발달장애 및 자폐증 아이를 치료하면서 교육할 수 있는 다양한 요육프로그램을 개발했다.', 
                'ko_tokens' : ['1990', '년', '설립', '된', '특수', '요', '육', '원', '은', '뇌', '발달', '을', '연구', '해', '만', '5', '세', '이하', '발달', '장애', '및', '자폐증', '아이', '를', '치료', '하', '면서', '교육', '할', '수', '있', '는', '다양', '한', '요육', '프로그램', '을', '개발', '했', '다', '.']
                },
                { # example of the 2nd sentence in the 1st article
                    ...
                },
                ... 
        	]
	},
    { # example of the 2nd article
        ...
    },
	... 
]

The data set is divided into three:
1) ai_hub_train_corpus.json (merged version of 2 and 3) [9844 pairs]
2) ai_hub_metric_eval_corpus [1500 pairs]
3) ai_hub_train_corpus_small [8344 pairs]

