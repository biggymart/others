# ibm hangul recognition in keras
# update: 2021. 11. 05.
# 한글 분류인데 한국어로 .md가 제대로 된 한국어로 작성되어야 하지 않겠니? ibm아?

원래 소스 코드는 다음 링크 참고
src: https://github.com/IBM/tensorflow-hangul-recognition
설명한 유투브 영상은 아래 참고
YouTube: https://www.youtube.com/watch?v=iefYaCOz00s

원래 코드는 텐서플로 버전 1으로 작성되서 가독성도 낮고 이해하기 어려움. 잘 돌아가지도 않는 거 같음. 하... 그래서 울며겨자먹기로 케라스 형식으로 다시 재해석함. 도움 되길 바람.
The original code has been written in tensorflow version 1, which my team found to be dull to understand. (Plus, for some unknown reasons, it seem the code is not so compatible with the current tensorflow version. You can find the original code in the folder '/retired') So, we decided to interpret the overall framework of the code and convert it to a keras code for enhanced clarity.

=====
코드 사용법
1. /fonts 폴더에 대충 구할 수 있는 폰트들 인터넷에서 긁어모을 것 (확장자 .ttf로 고를 것). 저작권 없는 거 사용하는 거 추천. (참고: 원래 코드에서는 폰트 40개 사용함) 
2. /tools 폴더 들어가보면 img_gen_xxx.py 있을 거임. xxx에는 생성하고 싶은 한글 글자 개수임. phoneme은 "음소"인데 자음 + 모음임 (비추). 원하는 거 실행시키셈. (참고: 원래 코드는 2350개 선택, 320,000개 이미지 생성함) 
=> 짤막한 원리 설명: /labels 폴더 들어가면 .txt 파일 있을텐데, 거기에 한줄한줄 읽어와서 /image-data 폴더에 (1) 해당 이미지 만들어주고 (2) .csv 파일에 '파일경로, 글자' 정보 써줌. 이미지 만들어줄 때는 DISTORTION_COUNT 만큼 추가로 변형된 이미지 만들어줌. (ex> DISTORTION_COUNT=3이면 각 글자당 4개 이미지 생성됨)
3. ibm_hangul_2350.py 실행하셈. /saved-model 폴더 만들어주고 .h5 파일 만들어 줄 것임.
4. predict_hangul_2350.py 실행하셈.

=====
